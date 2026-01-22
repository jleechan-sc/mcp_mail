"""Minimal MCP Agent Mail server with 8 core messaging tools."""

import asyncio
import functools
import inspect
import logging
import time
from collections import defaultdict
from contextlib import asynccontextmanager, suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, cast
from functools import wraps

from fastmcp import Context, FastMCP
from sqlalchemy import desc, func, select, update
from sqlalchemy.exc import IntegrityError, NoResultFound
from sqlalchemy.ext.asyncio import AsyncSession

from .config import Settings, get_settings
from .db import ensure_schema, get_session, init_engine
from .models import Agent, Message, MessageRecipient, Project
from .storage import (
    ProjectArchive,
    ensure_archive,
    write_message_bundle,
)
from .utils import sanitize_agent_name, slugify

logger = logging.getLogger(__name__)

TOOL_METRICS: defaultdict[str, dict[str, int]] = defaultdict(lambda: {"calls": 0, "errors": 0})

CLUSTER_SETUP = "infrastructure"
CLUSTER_IDENTITY = "identity"
CLUSTER_MESSAGING = "messaging"


class ToolExecutionError(Exception):
    """Tool execution error with structured error information."""

    def __init__(
        self, error_type: str, message: str, *, recoverable: bool = True, data: Optional[dict[str, Any]] = None
    ):
        super().__init__(message)
        self.error_type = error_type
        self.recoverable = recoverable
        self.data = data or {}

    def to_payload(self) -> dict[str, Any]:
        return {
            "error": {
                "type": self.error_type,
                "message": str(self),
                "recoverable": self.recoverable,
                "data": self.data,
            }
        }


def _record_tool_error(tool_name: str, exc: Exception) -> None:
    """Record tool error for metrics."""
    logger.warning(
        "tool_error",
        extra={
            "tool": tool_name,
            "error": type(exc).__name__,
            "error_message": str(exc),
        },
    )


def _instrument_tool(
    tool_name: str,
    *,
    cluster: str,
    capabilities: Optional[set[str]] = None,
    complexity: str = "medium",
    agent_arg: Optional[str] = None,
    project_arg: Optional[str] = None,
):
    """Decorator to instrument tool calls with metrics and error handling."""

    def decorator(func):
        signature = inspect.signature(func)

        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            metrics = TOOL_METRICS[tool_name]
            metrics["calls"] += 1

            result = None
            error = None
            try:
                result = await func(*args, **kwargs)
            except ToolExecutionError as exc:
                metrics["errors"] += 1
                _record_tool_error(tool_name, exc)
                error = exc
                raise
            except NoResultFound as exc:
                metrics["errors"] += 1
                _record_tool_error(tool_name, exc)
                wrapped_exc = ToolExecutionError(
                    "NOT_FOUND",
                    str(exc),
                    recoverable=True,
                    data={"tool": tool_name},
                )
                error = wrapped_exc
                raise wrapped_exc from exc
            except Exception as exc:
                metrics["errors"] += 1
                _record_tool_error(tool_name, exc)
                wrapped_exc = ToolExecutionError(
                    "UNHANDLED_EXCEPTION",
                    "Server encountered an unexpected error while executing tool.",
                    recoverable=False,
                    data={"tool": tool_name, "original_error": type(exc).__name__},
                )
                error = wrapped_exc
                raise wrapped_exc from exc

            return result

        with suppress(Exception):
            wrapper.__annotations__ = getattr(func, "__annotations__", {})
        return wrapper

    return decorator


def _lifespan_factory(settings: Settings):
    """Create lifespan context manager for FastMCP app."""

    @asynccontextmanager
    async def lifespan(app: FastMCP):
        init_engine(settings)
        await ensure_schema(settings)
        yield

    return lifespan


def _iso(dt: Any) -> str:
    """Return ISO-8601 in UTC from datetime or best-effort from string."""
    try:
        if isinstance(dt, str):
            try:
                parsed = datetime.fromisoformat(dt)
                return parsed.astimezone(timezone.utc).isoformat()
            except Exception:
                return dt
        if hasattr(dt, "astimezone"):
            return dt.astimezone(timezone.utc).isoformat()
        return str(dt)
    except Exception:
        return str(dt)


def _ensure_utc(dt: Optional[datetime]) -> Optional[datetime]:
    """Return a timezone-aware UTC datetime."""
    if dt is None:
        return None
    if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


# ============================================================================
# Database helper functions
# ============================================================================


async def _ensure_project(human_key: str) -> Project:
    """Idempotently create or get project by human_key."""
    slug = slugify(human_key)
    async with get_session() as session:
        result = await session.execute(select(Project).where(Project.slug == slug))
        project = result.scalar_one_or_none()
        if project:
            return project
        project = Project(slug=slug, human_key=human_key)
        session.add(project)
        await session.commit()
        await session.refresh(project)
        return project


async def _get_project_by_identifier(identifier: str) -> Project:
    """Get project by slug or human_key."""
    slug = slugify(identifier)
    async with get_session() as session:
        result = await session.execute(select(Project).where(Project.slug == slug))
        project = result.scalar_one_or_none()
        if not project:
            result = await session.execute(select(Project).where(Project.human_key == identifier))
            project = result.scalar_one_or_none()
        if not project:
            raise NoResultFound(f"Project not found for identifier: {identifier}")
        return project


async def _get_agent_by_name(name: str) -> Agent:
    """Get agent by name (globally unique)."""
    normalized = sanitize_agent_name(name)
    async with get_session() as session:
        result = await session.execute(
            select(Agent).where(
                func.lower(Agent.name) == normalized.lower(),
                Agent.is_active == True
            )
        )
        agent = result.scalar_one_or_none()
        if not agent:
            raise NoResultFound(f"Agent not found: {name}")
        return agent


async def _get_agent(project: Project, name: str) -> Agent:
    """Get agent by name within a project context."""
    agent = await _get_agent_by_name(name)
    # Verify agent belongs to this project
    if agent.project_id != project.id:
        raise NoResultFound(f"Agent '{name}' not found in project '{project.human_key}'")
    return agent


async def _get_or_create_agent(
    project: Project,
    name: Optional[str],
    program: str,
    model: str,
    task_description: str,
    settings: Settings,
    force_reclaim: bool = False,
) -> Agent:
    """Get or create agent with the given name."""
    if name:
        normalized = sanitize_agent_name(name)
    else:
        # Generate unique name
        from .utils import generate_agent_name

        normalized = generate_agent_name()

    now = datetime.now(timezone.utc)

    async with get_session() as session:
        # Check if agent exists
        result = await session.execute(select(Agent).where(func.lower(Agent.name) == normalized.lower()))
        existing = result.scalar_one_or_none()

        if existing:
            # Update existing agent
            existing.program = program
            existing.model = model
            existing.task_description = task_description
            existing.last_active_ts = now
            existing.project_id = project.id
            session.add(existing)
            await session.commit()
            await session.refresh(existing)
            return existing

        # Create new agent
        agent = Agent(
            name=normalized,
            program=program,
            model=model,
            task_description=task_description,
            project_id=project.id,
            inception_ts=now,
            last_active_ts=now,
        )
        session.add(agent)
        try:
            await session.commit()
            await session.refresh(agent)
            return agent
        except IntegrityError:
            await session.rollback()
            # Race condition - try to fetch again
            result = await session.execute(select(Agent).where(func.lower(Agent.name) == normalized.lower()))
            agent = result.scalar_one()
            return agent


async def _create_message(
    project: Project,
    sender: Agent,
    subject: str,
    body_md: str,
    recipient_records: list[tuple[Agent, str]],
    importance: str,
    ack_required: bool,
    thread_id: Optional[str],
    attachments_meta: list[dict[str, Any]],
) -> Message:
    """Create message and recipient records in database."""
    now = datetime.now(timezone.utc)

    async with get_session() as session:
        message = Message(
            project_id=project.id,
            sender_id=sender.id,
            subject=subject,
            body_md=body_md,
            created_ts=now,
            importance=importance,
            ack_required=ack_required,
            thread_id=thread_id,
        )
        session.add(message)
        await session.flush()

        # Create recipient records
        for recipient, recipient_type in recipient_records:
            rec = MessageRecipient(
                message_id=message.id,
                agent_id=recipient.id,
                kind=recipient_type,
            )
            session.add(rec)

        # Update sender's last_active_ts
        await session.execute(
            update(Agent).where(Agent.id == sender.id).values(last_active_ts=now)
        )

        await session.commit()
        await session.refresh(message)
        return message


def _message_frontmatter(
    message: Message,
    project: Project,
    sender: Agent,
    to_agents: list[Agent],
    cc_agents: list[Agent],
    bcc_agents: list[Agent],
    attachments_meta: list[dict[str, Any]],
) -> dict[str, Any]:
    """Generate message frontmatter for archive."""
    return {
        "message_id": str(message.id),
        "project": project.human_key,
        "from": sender.name,
        "to": [a.name for a in to_agents],
        "cc": [a.name for a in cc_agents],
        "bcc": [a.name for a in bcc_agents],
        "subject": message.subject,
        "created": _iso(message.created_ts),
        "importance": message.importance,
        "ack_required": message.ack_required,
        "thread_id": message.thread_id,
        "attachments": attachments_meta,
    }


def _message_to_dict(message: Message) -> dict[str, Any]:
    """Convert message to dictionary."""
    return {
        "id": message.id,
        "project_id": message.project_id,
        "sender_id": message.sender_id,
        "subject": message.subject,
        "body_md": message.body_md,
        "created_ts": _iso(message.created_ts),
        "importance": message.importance,
        "ack_required": message.ack_required,
        "thread_id": message.thread_id,
    }


def _project_to_dict(project: Project) -> dict[str, Any]:
    """Convert project to dictionary."""
    return {
        "id": project.id,
        "slug": project.slug,
        "human_key": project.human_key,
        "created_at": _iso(project.created_at),
    }


def _agent_to_dict(agent: Agent) -> dict[str, Any]:
    """Convert agent to dictionary."""
    return {
        "id": agent.id,
        "name": agent.name,
        "program": agent.program,
        "model": agent.model,
        "task_description": agent.task_description,
        "inception_ts": _iso(agent.inception_ts),
        "last_active_ts": _iso(agent.last_active_ts),
        "project_id": agent.project_id,
    }


async def _list_inbox(
    project: Project,
    agent: Agent,
    limit: int,
    urgent_only: bool = False,
    include_bodies: bool = False,
    since_ts: Optional[str] = None,
) -> list[dict[str, Any]]:
    """List inbox messages for an agent."""
    async with get_session() as session:
        query = (
            select(Message, MessageRecipient, Agent)
            .join(MessageRecipient, Message.id == MessageRecipient.message_id)
            .join(Agent, Message.sender_id == Agent.id)
            .where(
                MessageRecipient.agent_id == agent.id,
                Message.project_id == project.id,
            )
        )

        if urgent_only:
            query = query.where(Message.importance == "urgent")

        if since_ts:
            try:
                since_dt = datetime.fromisoformat(since_ts)
                # Ensure timezone-aware datetime for comparison
                if since_dt.tzinfo is None:
                    since_dt = since_dt.replace(tzinfo=timezone.utc)
                query = query.where(Message.created_ts >= since_dt)
            except ValueError:
                pass

        query = query.order_by(desc(Message.created_ts)).limit(limit)

        result = await session.execute(query)
        rows = result.all()

        items = []
        for message, recipient, sender in rows:
            item = {
                "id": message.id,
                "from": sender.name,
                "subject": message.subject,
                "created_ts": _iso(message.created_ts),
                "importance": message.importance,
                "ack_required": message.ack_required,
                "thread_id": message.thread_id,
                "read_ts": _iso(recipient.read_ts) if recipient.read_ts else None,
                "ack_ts": _iso(recipient.ack_ts) if recipient.ack_ts else None,
            }
            if include_bodies:
                item["body_md"] = message.body_md
            items.append(item)

        return items


def build_mcp_server() -> FastMCP:
    """Create and configure the minimal FastMCP server with 8 core tools."""
    settings: Settings = get_settings()
    lifespan = _lifespan_factory(settings)

    instructions = (
        "You are the MCP Agent Mail coordination server. "
        "Provide message routing and coordination tooling to cooperating agents."
    )

    mcp = FastMCP(name="mcp-agent-mail", instructions=instructions, lifespan=lifespan)

    # ========================================================================
    # Tool 1: ensure_project
    # ========================================================================
    @mcp.tool(name="ensure_project")
    @_instrument_tool(
        "ensure_project",
        cluster=CLUSTER_SETUP,
        capabilities={"infrastructure", "storage"},
        complexity="low",
        project_arg="human_key",
    )
    async def ensure_project(ctx: Context, human_key: str) -> dict[str, Any]:
        """
        Idempotently create or ensure a project exists for the given human key.

        When to use
        -----------
        - First call in a workflow targeting a new repo/path/project identifier.
        - As a guard before registering agents or sending messages.

        Parameters
        ----------
        human_key : str
            Any string identifier for the project (e.g., "/path/to/repo", "project-name").

        Returns
        -------
        dict
            Minimal project descriptor: { id, slug, human_key, created_at }.
        """
        await ctx.info(f"Ensuring project for key '{human_key}'.")
        project = await _ensure_project(human_key)
        await ensure_archive(settings, project.slug)
        return _project_to_dict(project)

    # ========================================================================
    # Tool 2: register_agent
    # ========================================================================
    @mcp.tool(name="register_agent")
    @_instrument_tool(
        "register_agent",
        cluster=CLUSTER_IDENTITY,
        capabilities={"identity"},
        agent_arg="name",
        project_arg="project_key",
    )
    async def register_agent(
        ctx: Context,
        project_key: str,
        program: str,
        model: str,
        name: Optional[str] = None,
        task_description: str = "",
        attachments_policy: str = "auto",
        force_reclaim: bool = False,
    ) -> dict[str, Any]:
        """
        Create or update an agent identity within a project.

        When to use
        -----------
        - At the start of a coding session by any automated agent.
        - To update an existing agent's program/model/task metadata.

        Parameters
        ----------
        project_key : str
            Project identifier (will be auto-created if doesn't exist).
        program : str
            The agent program (e.g., "claude-code", "cursor").
        model : str
            The underlying model (e.g., "claude-sonnet-4.5").
        name : Optional[str]
            Agent name. If omitted, auto-generated.
        task_description : str
            Short description of current focus.
        attachments_policy : str
            Attachment handling: "auto", "inline", or "file".
        force_reclaim : bool
            If True, forcefully reclaim this agent name.

        Returns
        -------
        dict
            Agent profile: { id, name, program, model, task_description, inception_ts, last_active_ts, project_id }
        """
        project = await _ensure_project(project_key)
        await ensure_archive(settings, project.slug)

        ap = (attachments_policy or "auto").lower()
        if ap not in {"auto", "inline", "file"}:
            ap = "auto"

        agent = await _get_or_create_agent(
            project, name, program, model, task_description, settings, force_reclaim=force_reclaim
        )

        # Update attachments policy
        if getattr(agent, "attachments_policy", None) != ap:
            async with get_session() as session:
                db_agent = await session.get(Agent, agent.id)
                if db_agent:
                    db_agent.attachments_policy = ap
                    session.add(db_agent)
                    await session.commit()
                    await session.refresh(db_agent)
                    agent = db_agent

        await ctx.info(f"Registered agent '{agent.name}' for project '{project.human_key}'.")
        return _agent_to_dict(agent)

    # ========================================================================
    # Tool 3: get_agent_info
    # ========================================================================
    @mcp.tool(name="get_agent_info")
    @_instrument_tool(
        "get_agent_info",
        cluster=CLUSTER_IDENTITY,
        capabilities={"identity"},
        project_arg="project_key",
        agent_arg="agent_name",
    )
    async def get_agent_info(
        ctx: Context,
        project_key: str,
        agent_name: str,
    ) -> dict[str, Any]:
        """
        Get detailed information about an agent.

        Parameters
        ----------
        project_key : str
            Project identifier.
        agent_name : str
            Agent name to look up.

        Returns
        -------
        dict
            Agent profile with all metadata.
        """
        project = await _get_project_by_identifier(project_key)
        agent = await _get_agent(project, agent_name)
        await ctx.info(f"Retrieved info for agent '{agent_name}' in project '{project.human_key}'.")
        return _agent_to_dict(agent)

    # ========================================================================
    # Tool 4: send_message
    # ========================================================================
    @mcp.tool(name="send_message")
    @_instrument_tool(
        "send_message",
        cluster=CLUSTER_MESSAGING,
        capabilities={"messaging", "write"},
        project_arg="project_key",
        agent_arg="sender_name",
    )
    async def send_message(
        ctx: Context,
        project_key: str,
        sender_name: str,
        to: list[str],
        subject: str,
        body_md: str,
        cc: Optional[list[str]] = None,
        bcc: Optional[list[str]] = None,
        importance: str = "normal",
        ack_required: bool = False,
        thread_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Send a Markdown message to one or more recipients.

        Parameters
        ----------
        project_key : str
            Project identifier.
        sender_name : str
            Name of the sending agent.
        to : list[str]
            List of recipient agent names.
        subject : str
            Message subject.
        body_md : str
            Message body in Markdown format.
        cc : Optional[list[str]]
            CC recipients.
        bcc : Optional[list[str]]
            BCC recipients.
        importance : str
            Message importance: "normal" or "urgent".
        ack_required : bool
            Whether acknowledgment is required.
        thread_id : Optional[str]
            Thread ID for message threading.

        Returns
        -------
        dict
            Delivery confirmation with message details.
        """
        if not to and not cc and not bcc:
            raise ValueError("At least one recipient must be specified.")

        project = await _get_project_by_identifier(project_key)
        sender = await _get_agent(project, sender_name)

        # Deduplicate recipients
        def _unique(items: list[str]) -> list[str]:
            seen: set[str] = set()
            ordered: list[str] = []
            for item in items or []:
                if item not in seen:
                    seen.add(item)
                    ordered.append(item)
            return ordered

        to = _unique(to or [])
        cc = _unique(cc or [])
        bcc = _unique(bcc or [])

        # Fetch recipient agents
        to_agents = [await _get_agent_by_name(name) for name in to]
        cc_agents = [await _get_agent_by_name(name) for name in cc]
        bcc_agents = [await _get_agent_by_name(name) for name in bcc]

        recipient_records: list[tuple[Agent, str]] = [(agent, "to") for agent in to_agents]
        recipient_records.extend((agent, "cc") for agent in cc_agents)
        recipient_records.extend((agent, "bcc") for agent in bcc_agents)

        archive = await ensure_archive(settings, project.slug)

        # Create message
        message = await _create_message(
            project,
            sender,
            subject,
            body_md,
            recipient_records,
            importance,
            ack_required,
            thread_id,
            [],  # No attachments in minimal version
        )

        # Write to archive
        frontmatter = _message_frontmatter(
            message,
            project,
            sender,
            to_agents,
            cc_agents,
            bcc_agents,
            [],
        )

        recipients_for_archive = [agent.name for agent in to_agents + cc_agents + bcc_agents]
        await write_message_bundle(
            archive,
            frontmatter,
            body_md,
            sender.name,
            recipients_for_archive,
            [],  # No attachment files
        )

        payload = _message_to_dict(message)
        payload.update(
            {
                "from": sender.name,
                "to": [agent.name for agent in to_agents],
                "cc": [agent.name for agent in cc_agents],
                "bcc": [agent.name for agent in bcc_agents],
            }
        )

        await ctx.info(f"Sent message from '{sender_name}' to {len(recipient_records)} recipients.")
        return {
            "deliveries": [
                {
                    "project": project.human_key,
                    "payload": payload,
                }
            ],
            "count": 1,
        }

    # ========================================================================
    # Tool 5: fetch_inbox
    # ========================================================================
    @mcp.tool(name="fetch_inbox")
    @_instrument_tool(
        "fetch_inbox",
        cluster=CLUSTER_MESSAGING,
        capabilities={"messaging", "read"},
        project_arg="project_key",
        agent_arg="agent_name",
    )
    async def fetch_inbox(
        ctx: Context,
        project_key: str,
        agent_name: str,
        limit: int = 10,
        urgent_only: bool = False,
        include_bodies: bool = False,
        since_ts: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Fetch inbox messages for an agent.

        Parameters
        ----------
        project_key : str
            Project identifier.
        agent_name : str
            Agent name to fetch inbox for.
        limit : int
            Maximum number of messages to return (default: 10).
        urgent_only : bool
            Only return urgent messages.
        include_bodies : bool
            Include message bodies in response.
        since_ts : Optional[str]
            Only return messages after this timestamp (ISO format).

        Returns
        -------
        dict
            Inbox summary with messages list.
        """
        project = await _get_project_by_identifier(project_key)
        agent = await _get_agent(project, agent_name)

        items = await _list_inbox(
            project,
            agent,
            limit,
            urgent_only=urgent_only,
            include_bodies=include_bodies,
            since_ts=since_ts,
        )

        await ctx.info(f"Fetched {len(items)} inbox messages for '{agent_name}'.")
        return {
            "agent": agent.name,
            "project": project.human_key,
            "messages": items,
            "count": len(items),
        }

    # ========================================================================
    # Tool 6: reply_message
    # ========================================================================
    @mcp.tool(name="reply_message")
    @_instrument_tool(
        "reply_message",
        cluster=CLUSTER_MESSAGING,
        capabilities={"messaging", "write"},
        project_arg="project_key",
        agent_arg="sender_name",
    )
    async def reply_message(
        ctx: Context,
        project_key: str,
        sender_name: str,
        original_message_id: int,
        body_md: str,
        importance: str = "normal",
        ack_required: bool = False,
    ) -> dict[str, Any]:
        """
        Reply to an existing message.

        Parameters
        ----------
        project_key : str
            Project identifier.
        sender_name : str
            Name of the sending agent.
        original_message_id : int
            ID of the message being replied to.
        body_md : str
            Reply body in Markdown format.
        importance : str
            Message importance: "normal" or "urgent".
        ack_required : bool
            Whether acknowledgment is required.

        Returns
        -------
        dict
            Delivery confirmation with message details.
        """
        project = await _get_project_by_identifier(project_key)
        sender = await _get_agent(project, sender_name)

        # Fetch original message
        async with get_session() as session:
            result = await session.execute(
                select(Message, Agent)
                .join(Agent, Message.sender_id == Agent.id)
                .where(Message.id == original_message_id, Message.project_id == project.id)
            )
            row = result.one_or_none()
            if not row:
                raise NoResultFound(f"Original message {original_message_id} not found.")
            original_message, original_sender = row

        # Reply to original sender
        subject = f"Re: {original_message.subject}"
        thread_id = original_message.thread_id or str(original_message.id)

        archive = await ensure_archive(settings, project.slug)

        recipient_records = [(original_sender, "to")]

        message = await _create_message(
            project,
            sender,
            subject,
            body_md,
            recipient_records,
            importance,
            ack_required,
            thread_id,
            [],
        )

        frontmatter = _message_frontmatter(
            message,
            project,
            sender,
            [original_sender],
            [],
            [],
            [],
        )

        await write_message_bundle(
            archive,
            frontmatter,
            body_md,
            sender.name,
            [original_sender.name],
            [],
        )

        payload = _message_to_dict(message)
        payload.update(
            {
                "from": sender.name,
                "to": [original_sender.name],
                "cc": [],
                "bcc": [],
            }
        )

        await ctx.info(f"Sent reply from '{sender_name}' to '{original_sender.name}'.")
        return {
            "deliveries": [
                {
                    "project": project.human_key,
                    "payload": payload,
                }
            ],
            "count": 1,
        }

    # ========================================================================
    # Tool 7: mark_message_read
    # ========================================================================
    @mcp.tool(name="mark_message_read")
    @_instrument_tool(
        "mark_message_read",
        cluster=CLUSTER_MESSAGING,
        capabilities={"messaging", "write"},
        project_arg="project_key",
        agent_arg="agent_name",
    )
    async def mark_message_read(
        ctx: Context,
        project_key: str,
        agent_name: str,
        message_id: int,
    ) -> dict[str, Any]:
        """
        Mark a message as read.

        Parameters
        ----------
        project_key : str
            Project identifier.
        agent_name : str
            Agent name marking the message as read.
        message_id : int
            Message ID to mark as read.

        Returns
        -------
        dict
            Confirmation with updated read timestamp.
        """
        project = await _get_project_by_identifier(project_key)
        agent = await _get_agent(project, agent_name)

        now = datetime.now(timezone.utc)

        async with get_session() as session:
            result = await session.execute(
                select(MessageRecipient)
                .join(Message, MessageRecipient.message_id == Message.id)
                .where(
                    MessageRecipient.message_id == message_id,
                    MessageRecipient.agent_id == agent.id,
                    Message.project_id == project.id,
                )
            )
            recipient = result.scalar_one_or_none()
            if not recipient:
                raise NoResultFound(f"Message {message_id} not found in inbox for agent '{agent_name}'.")

            if not recipient.read_ts:
                recipient.read_ts = now
                session.add(recipient)
                await session.commit()

        await ctx.info(f"Marked message {message_id} as read for '{agent_name}'.")
        return {
            "message_id": message_id,
            "agent": agent_name,
            "read_ts": _iso(now),
        }

    # ========================================================================
    # Tool 8: acknowledge_message
    # ========================================================================
    @mcp.tool(name="acknowledge_message")
    @_instrument_tool(
        "acknowledge_message",
        cluster=CLUSTER_MESSAGING,
        capabilities={"messaging", "write"},
        project_arg="project_key",
        agent_arg="agent_name",
    )
    async def acknowledge_message(
        ctx: Context,
        project_key: str,
        agent_name: str,
        message_id: int,
    ) -> dict[str, Any]:
        """
        Acknowledge a message (marks both read and acknowledged).

        Parameters
        ----------
        project_key : str
            Project identifier.
        agent_name : str
            Agent name acknowledging the message.
        message_id : int
            Message ID to acknowledge.

        Returns
        -------
        dict
            Confirmation with updated read and ack timestamps.
        """
        project = await _get_project_by_identifier(project_key)
        agent = await _get_agent(project, agent_name)

        now = datetime.now(timezone.utc)

        async with get_session() as session:
            result = await session.execute(
                select(MessageRecipient)
                .join(Message, MessageRecipient.message_id == Message.id)
                .where(
                    MessageRecipient.message_id == message_id,
                    MessageRecipient.agent_id == agent.id,
                    Message.project_id == project.id,
                )
            )
            recipient = result.scalar_one_or_none()
            if not recipient:
                raise NoResultFound(f"Message {message_id} not found in inbox for agent '{agent_name}'.")

            if not recipient.read_ts:
                recipient.read_ts = now
            if not recipient.ack_ts:
                recipient.ack_ts = now
            session.add(recipient)
            await session.commit()

        await ctx.info(f"Acknowledged message {message_id} for '{agent_name}'.")
        return {
            "message_id": message_id,
            "agent": agent_name,
            "read_ts": _iso(recipient.read_ts or now),
            "ack_ts": _iso(now),
        }

    return mcp


def create_app() -> FastMCP:
    """Create the FastMCP application instance."""
    return build_mcp_server()
