"""Browser control tools for Karasu voice agent using Playwright."""

import asyncio
import contextlib
import logging
import re
import tempfile
import time
from pathlib import Path

from livekit.agents import RunContext, function_tool
from livekit.agents.llm import ToolError
from playwright.async_api import (
    Error as PlaywrightError,
)
from playwright.async_api import (
    Page,
    Playwright,
    async_playwright,
)

logger = logging.getLogger("browser_tools")


class BrowserManager:
    """Manages a Playwright Chromium browser with multi-tab support."""

    def __init__(self) -> None:
        self._playwright: Playwright | None = None
        self._browser = None
        self._pages: list[Page] = []
        self._active_index: int = 0

    async def start(self) -> None:
        if self._browser:
            return
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(headless=False)
        page = await self._browser.new_page()
        self._pages = [page]
        self._active_index = 0
        logger.info("Browser launched")

    async def close(self) -> None:
        if self._browser:
            await self._browser.close()
            self._browser = None
            self._pages = []
            self._active_index = 0
        if self._playwright:
            await self._playwright.stop()
            self._playwright = None
        logger.info("Browser closed")

    @property
    def active_page(self) -> Page:
        if not self._pages:
            raise ToolError("No open tabs.")
        return self._pages[self._active_index]

    @property
    def page_count(self) -> int:
        return len(self._pages)

    @property
    def active_index(self) -> int:
        return self._active_index

    async def ensure_started(self) -> None:
        if not self._browser:
            await self.start()

    async def new_page(self, url: str | None = None) -> Page:
        if not self._browser:
            raise ToolError("Browser is not running.")
        page = await self._browser.new_page()
        if url:
            try:
                await page.goto(url, wait_until="domcontentloaded", timeout=30000)
            except PlaywrightError as e:
                await page.close()
                raise ToolError(f"Failed to navigate to {url}: {e}") from e
        self._pages.append(page)
        self._active_index = len(self._pages) - 1
        return page

    async def close_page(self, index: int | None = None) -> None:
        if not self._pages:
            raise ToolError("No open tabs.")
        idx = index if index is not None else self._active_index
        if idx < 0 or idx >= len(self._pages):
            raise ToolError(
                f"Invalid tab index {idx}. Tabs are numbered 0 to {len(self._pages) - 1}."
            )
        page = self._pages.pop(idx)
        await page.close()
        if not self._pages:
            page = await self._browser.new_page()
            self._pages = [page]
            self._active_index = 0
        elif self._active_index >= len(self._pages):
            self._active_index = len(self._pages) - 1

    async def switch_to(self, index: int) -> None:
        if index < 0 or index >= len(self._pages):
            raise ToolError(
                f"Invalid tab index {index}. Tabs are numbered 0 to {len(self._pages) - 1}."
            )
        self._active_index = index


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_manager: BrowserManager | None = None


async def get_manager() -> BrowserManager:
    global _manager
    if _manager is None:
        _manager = BrowserManager()
    await _manager.ensure_started()
    return _manager


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@function_tool
async def browser_new_tab(context: RunContext, url: str = "") -> str:
    """Open a new browser tab. Optionally navigate to a URL.

    Args:
        url: The URL to navigate to. Leave empty to open a blank tab.
    """
    manager = await get_manager()
    await manager.new_page(url if url else None)
    title = await manager.active_page.title() if url else "Blank page"
    return (
        f"Opened tab {manager.active_index}: {title}. Total tabs: {manager.page_count}"
    )


@function_tool
async def browser_switch_tab(context: RunContext, index: int) -> str:
    """Switch to a different browser tab by index (0-based).

    Args:
        index: The tab index to switch to.
    """
    manager = await get_manager()
    await manager.switch_to(index)
    page = manager.active_page
    title = await page.title()
    url = page.url
    return f"Switched to tab {index}: {title} ({url})"


@function_tool
async def browser_close_tab(context: RunContext, index: int = -1) -> str:
    """Close a browser tab. Defaults to the current tab.

    Args:
        index: The tab index to close. Use -1 for the current tab.
    """
    manager = await get_manager()
    close_idx = index if index >= 0 else manager.active_index
    await manager.close_page(close_idx)
    return (
        f"Closed tab. Now on tab {manager.active_index}. "
        f"Total tabs: {manager.page_count}"
    )


@function_tool
async def browser_open(context: RunContext, url: str) -> str:
    """Navigate the current browser tab to a URL.

    Args:
        url: The URL to navigate to.
    """
    context.disallow_interruptions()
    manager = await get_manager()
    page = manager.active_page
    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=30000)
    except PlaywrightError as e:
        raise ToolError(f"Failed to navigate to {url}: {e}") from e
    title = await page.title()
    return f"Navigated to {url}. Page title: {title}"


@function_tool
async def browser_go_back(context: RunContext) -> str:
    """Go back to the previous page in browser history."""
    context.disallow_interruptions()
    manager = await get_manager()
    page = manager.active_page
    try:
        await page.go_back(timeout=15000)
    except PlaywrightError as e:
        raise ToolError(f"Could not go back: {e}") from e
    title = await page.title()
    url = page.url
    return f"Went back. Now on: {title} ({url})"


@function_tool
async def browser_go_forward(context: RunContext) -> str:
    """Go forward in browser history."""
    context.disallow_interruptions()
    manager = await get_manager()
    page = manager.active_page
    try:
        await page.go_forward(timeout=15000)
    except PlaywrightError as e:
        raise ToolError(f"Could not go forward: {e}") from e
    title = await page.title()
    url = page.url
    return f"Went forward. Now on: {title} ({url})"


@function_tool
async def browser_reload(context: RunContext) -> str:
    """Reload the current page."""
    context.disallow_interruptions()
    manager = await get_manager()
    page = manager.active_page
    try:
        await page.reload(timeout=15000)
    except PlaywrightError as e:
        raise ToolError(f"Could not reload: {e}") from e
    title = await page.title()
    return f"Reloaded page: {title}"


@function_tool
async def browser_read_page(context: RunContext, max_chars: int = 3000) -> str:
    """Read the visible text content of the current page. Use this to understand what is on the page before interacting.

    Args:
        max_chars: Maximum characters to return. Default 3000.
    """
    manager = await get_manager()
    page = manager.active_page
    try:
        text = await page.inner_text("body")
    except PlaywrightError:
        text = ""
    text = re.sub(r"\n\s*\n", "\n\n", text)
    text = re.sub(r"  +", " ", text)
    text = text.strip()
    title = await page.title()
    url = page.url
    header = f"Page: {title}\nURL: {url}\n\n"
    if len(text) > max_chars:
        text = text[:max_chars] + "\n...[content truncated]"
    return header + (text or "Page appears to be empty.")


@function_tool
async def browser_find_interactive_elements(context: RunContext) -> str:
    """Find all interactive elements (links, buttons, inputs, etc.) on the current page. Returns a numbered list with suggested selectors for clicking or filling."""
    manager = await get_manager()
    page = manager.active_page
    elements = await page.evaluate("""() => {
        const sel = 'a, button, input, select, textarea, [role="button"], [role="link"], [role="tab"], [role="menuitem"], [onclick]';
        const els = document.querySelectorAll(sel);
        return Array.from(els).slice(0, 50).map((el, i) => {
            const tag = el.tagName.toLowerCase();
            const text = (el.textContent || '').trim().slice(0, 80);
            const placeholder = el.getAttribute('placeholder') || '';
            const inputType = el.getAttribute('type') || '';
            const href = el.getAttribute('href') || '';
            const id = el.id || '';
            const ariaLabel = el.getAttribute('aria-label') || '';

            let selector = '';
            if (id) selector = '#' + id;
            else if (text) selector = 'text="' + text.slice(0, 50) + '"';
            else if (placeholder) selector = tag + '[placeholder="' + placeholder + '"]';
            else if (ariaLabel) selector = 'text="' + ariaLabel + '"';
            else selector = tag + ':nth-of-type(' + (i + 1) + ')';

            let kind = tag;
            if (tag === 'input' && inputType) kind = inputType;

            const label = text || placeholder || ariaLabel;
            return { index: i + 1, kind, text: label.slice(0, 60), selector, href: href.slice(0, 100) };
        });
    }""")
    if not elements:
        return "No interactive elements found on this page."
    lines = ["Interactive elements on the page:"]
    for el in elements:
        desc = f"{el['index']}. [{el['kind']}]"
        if el["text"]:
            desc += f' "{el["text"]}"'
        if el["href"]:
            desc += f" -> {el['href']}"
        desc += f" -> selector: {el['selector']}"
        lines.append(desc)
    return "\n".join(lines)


@function_tool
async def browser_screenshot(context: RunContext) -> str:
    """Take a screenshot of the current page and save it to a file."""
    manager = await get_manager()
    page = manager.active_page
    filename = f"karasu_screenshot_{int(time.time())}.png"
    path = Path(tempfile.gettempdir()) / filename
    await page.screenshot(path=str(path), full_page=False)
    title = await page.title()
    return f"Screenshot saved to {path}. Page title: {title}"


@function_tool
async def browser_click(context: RunContext, selector: str) -> str:
    """Click an element on the page. Use CSS selectors (#id, .class, tag) or text selectors (text="Login").

    Args:
        selector: The Playwright-style selector for the element to click.
    """
    context.disallow_interruptions()
    manager = await get_manager()
    page = manager.active_page
    try:
        await page.locator(selector).click(timeout=10000)
    except PlaywrightError as e:
        raise ToolError(f"Could not click '{selector}': {e}") from e
    with contextlib.suppress(PlaywrightError):
        await page.wait_for_load_state("domcontentloaded", timeout=5000)
    title = await page.title()
    url = page.url
    return f"Clicked '{selector}'. Now on: {title} ({url})"


@function_tool
async def browser_fill(context: RunContext, selector: str, value: str) -> str:
    """Fill an input or textarea field with the given value (clears existing content first).

    Args:
        selector: The Playwright-style selector for the input field.
        value: The text to fill in.
    """
    context.disallow_interruptions()
    manager = await get_manager()
    page = manager.active_page
    try:
        await page.locator(selector).fill(value, timeout=10000)
    except PlaywrightError as e:
        raise ToolError(f"Could not fill '{selector}': {e}") from e
    return f"Filled '{selector}' with '{value}'"


@function_tool
async def browser_type(context: RunContext, selector: str, value: str) -> str:
    """Type text into an input field character by character. Use this for search boxes or autocomplete fields.

    Args:
        selector: The Playwright-style selector for the input field.
        value: The text to type.
    """
    context.disallow_interruptions()
    manager = await get_manager()
    page = manager.active_page
    try:
        await page.locator(selector).press_sequentially(value, timeout=30000)
    except PlaywrightError as e:
        raise ToolError(f"Could not type into '{selector}': {e}") from e
    return f"Typed '{value}' into '{selector}'"


@function_tool
async def browser_select(context: RunContext, selector: str, value: str) -> str:
    """Select an option from a dropdown menu.

    Args:
        selector: The Playwright-style selector for the select element.
        value: The value or label of the option to select.
    """
    context.disallow_interruptions()
    manager = await get_manager()
    page = manager.active_page
    try:
        await page.locator(selector).select_option(value, timeout=10000)
    except PlaywrightError as e:
        raise ToolError(f"Could not select option in '{selector}': {e}") from e
    return f"Selected '{value}' in '{selector}'"


@function_tool
async def browser_press(context: RunContext, key: str, selector: str = "") -> str:
    """Press a keyboard key. Common keys: Enter, Tab, Escape, ArrowDown, Backspace.

    Args:
        key: The key to press (e.g. "Enter", "Tab", "Escape").
        selector: Optional element to focus before pressing. If empty, presses on the page.
    """
    context.disallow_interruptions()
    manager = await get_manager()
    page = manager.active_page
    try:
        if selector:
            await page.locator(selector).press(key, timeout=10000)
        else:
            await page.keyboard.press(key)
    except PlaywrightError as e:
        raise ToolError(f"Could not press '{key}': {e}") from e
    return f"Pressed '{key}'"


@function_tool
async def browser_wait(context: RunContext, seconds: float) -> str:
    """Wait for a number of seconds.

    Args:
        seconds: Number of seconds to wait.
    """
    if seconds < 0:
        raise ToolError("Wait seconds must be non-negative.")
    await asyncio.sleep(seconds)
    return f"Waited {seconds} seconds."
