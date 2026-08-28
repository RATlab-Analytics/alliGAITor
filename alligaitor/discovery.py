"""Auto-discovery of a group's sessions from a folder of videos, using
regex-captured filename tokens for session identity and camera role.

Used by the GUI's config editor to build
:class:`alligaitor.config.SessionConfig` entries; see
:class:`alligaitor.config.DiscoveryConfig` for the rules this reads.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from alligaitor.config import CAMERA_ROLES, DiscoveryConfig, SessionConfig

PathLike = Union[str, Path]

VIDEO_EXTENSIONS = (".mp4",)


def find_videos(folder: PathLike) -> List[Path]:
    """Every video file under ``folder``, searched recursively, sorted."""
    folder = Path(folder)
    if not folder.exists():
        return []
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in VIDEO_EXTENSIONS)


def camera_tokens(videos: List[Path], camera_regex: str) -> List[str]:
    """Distinct camera tokens found across ``videos`` matching
    ``camera_regex``, in first-encounter order."""
    pattern = re.compile(camera_regex)
    tokens: List[str] = []
    seen = set()
    for video in videos:
        m = pattern.search(video.name)
        if m and m.group(1) not in seen:
            seen.add(m.group(1))
            tokens.append(m.group(1))
    return tokens


def representative_video_for_token(videos: List[Path], camera_regex: str, token: str) -> Optional[Path]:
    """First video whose ``camera_regex`` match equals ``token``."""
    pattern = re.compile(camera_regex)
    for video in videos:
        m = pattern.search(video.name)
        if m and m.group(1) == token:
            return video
    return None


def discover_sessions(
    discovery: DiscoveryConfig,
    cropped_dir: PathLike,
    predictions_dir: PathLike,
) -> Tuple[List[SessionConfig], List[str]]:
    """Group ``discovery.input_dir``'s videos into sessions using
    ``discovery.id_regex`` (session name) and ``discovery.camera_regex``
    (camera token, mapped to a role via ``discovery.camera_role_map``). A
    session is only emitted if it has exactly one video per role in
    :data:`alligaitor.config.CAMERA_ROLES`; anything else is reported in
    the second return value.

    Each emitted session's ``videos[role]`` points at
    ``<cropped_dir>/<path relative to discovery.input_dir>``, matching
    where the crop tool writes its output. ``output_dir`` is set to
    ``<predictions_dir>/<session name>``.

    Returns:
        ``(sessions, problems)``, where ``problems`` describes anything
        that couldn't be grouped.
    """
    id_pattern = re.compile(discovery.id_regex)
    camera_pattern = re.compile(discovery.camera_regex)
    cropped_dir = Path(cropped_dir)
    predictions_dir = Path(predictions_dir)

    # Guard against a regex with no capture group crashing at .group(1) below.
    if id_pattern.groups < 1:
        return [], [f"id regex has no capture group: {discovery.id_regex!r}"]
    if camera_pattern.groups < 1:
        return [], [f"camera regex has no capture group: {discovery.camera_regex!r}"]

    videos = find_videos(discovery.input_dir)
    problems: List[str] = []

    by_session: Dict[str, Dict[str, Path]] = {}
    collisions: Dict[str, Dict[str, List[Path]]] = {}

    for video in videos:
        id_match = id_pattern.search(video.name)
        camera_match = camera_pattern.search(video.name)
        if not id_match:
            problems.append(f"{video.name}: id regex didn't match")
            continue
        if not camera_match:
            problems.append(f"{video.name}: camera regex didn't match")
            continue

        session_name = id_match.group(1)
        token = camera_match.group(1)
        role = discovery.camera_role_map.get(token)
        if role is None:
            problems.append(f"{video.name}: camera token '{token}' isn't assigned to a role")
            continue
        if role not in CAMERA_ROLES:
            problems.append(f"{video.name}: '{role}' isn't a valid camera role {CAMERA_ROLES}")
            continue

        session_roles = by_session.setdefault(session_name, {})
        if role in session_roles:
            collisions.setdefault(session_name, {}).setdefault(role, [session_roles[role]]).append(video)
            continue
        session_roles[role] = video

    for session_name, roles in collisions.items():
        for role, paths in roles.items():
            names = ", ".join(p.name for p in paths)
            problems.append(f"session '{session_name}': more than one '{role}' video ({names})")

    sessions: List[SessionConfig] = []
    for session_name, roles in sorted(by_session.items()):
        if session_name in collisions:
            continue  # already reported above
        missing = [r for r in CAMERA_ROLES if r not in roles]
        if missing:
            have = ", ".join(f"{r}={p.name}" for r, p in roles.items()) or "none"
            problems.append(f"session '{session_name}': missing {missing} (have: {have})")
            continue

        cropped_videos = {
            role: cropped_dir / roles[role].relative_to(discovery.input_dir) for role in CAMERA_ROLES
        }
        rat_id = discovery.rat_id_overrides.get(session_name, session_name)
        sessions.append(
            SessionConfig(
                name=session_name,
                videos=cropped_videos,
                output_dir=predictions_dir / session_name,
                rat_id=rat_id,
            )
        )

    return sessions, problems
