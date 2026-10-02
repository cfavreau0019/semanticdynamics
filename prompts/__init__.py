"""
Versioned prompt templates, independent of any one task.

    from prompts import get_template

    t = get_template("celtic_cross_v2")          # pinned version; "celtic_cross" = latest
    p = t.render(cards=draw, persona=persona)    # persona is optional for this template
    p.messages, p.template_id, p.template_sha256

  template   PromptTemplate, RenderedPrompt, register_filter, PromptError
  library    PromptLibrary (TOML files + lock file), get_template, default_library
  __main__   python -m prompts list | show | diff | lock | verify | render
"""
from prompts.template import PromptError, PromptTemplate, RenderedPrompt, parse_id, register_filter
from prompts.library import (
    DRAFT, LIBRARY_DIR, LOCKED, MODIFIED, PromptLibrary, PromptLockError, default_library, get_template,
)

__all__ = [
    "PromptError", "PromptTemplate", "RenderedPrompt", "parse_id", "register_filter",
    "DRAFT", "LIBRARY_DIR", "LOCKED", "MODIFIED", "PromptLibrary", "PromptLockError", "default_library",
    "get_template",
]
