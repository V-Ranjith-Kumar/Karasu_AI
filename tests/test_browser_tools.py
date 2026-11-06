"""Tests for browser tools."""

import sys
from pathlib import Path

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from livekit.agents.llm import ToolError

import browser_tools
from browser_tools import (
    BrowserManager,
    browser_click,
    browser_close_tab,
    browser_fill,
    browser_find_interactive_elements,
    browser_go_back,
    browser_go_forward,
    browser_new_tab,
    browser_open,
    browser_press,
    browser_read_page,
    browser_reload,
    browser_screenshot,
    browser_select,
    browser_switch_tab,
    browser_type,
    browser_wait,
)


class MockRunContext:
    def disallow_interruptions(self):
        pass


# ---------------------------------------------------------------------------
# Test pages
# ---------------------------------------------------------------------------

INDEX_HTML = """
<html><head><title>Test Page</title></head>
<body>
<h1>Hello World</h1>
<p>This is a test page for Karasu browser tools.</p>
<a href="/target">Click me</a>
</body></html>
"""

LINKS_HTML = """
<html><head><title>Links Page</title></head>
<body>
<h1>Interactive Elements</h1>
<a id="link1" href="/target">First Link</a>
<a id="link2" href="/target">Second Link</a>
<button id="btn1">Button One</button>
<button id="btn2">Button Two</button>
<input type="text" id="search" placeholder="Search...">
</body></html>
"""

FORM_HTML = """
<html><head><title>Form Page</title></head>
<body>
<form id="testform">
    <input type="text" id="name" name="name" placeholder="Enter name">
    <select id="color" name="color">
        <option value="">Select a color</option>
        <option value="red">Red</option>
        <option value="blue">Blue</option>
        <option value="green">Green</option>
    </select>
    <textarea id="bio" name="bio" placeholder="Tell us about yourself"></textarea>
    <button type="submit" id="submit">Submit</button>
</form>
</body></html>
"""

TARGET_HTML = """
<html><head><title>Target Page</title></head>
<body>
<h1>You navigated here!</h1>
<p>This is the target page after clicking a link.</p>
</body></html>
"""


async def _handler_index(request):
    return web.Response(text=INDEX_HTML, content_type="text/html")


async def _handler_links(request):
    return web.Response(text=LINKS_HTML, content_type="text/html")


async def _handler_form(request):
    return web.Response(text=FORM_HTML, content_type="text/html")


async def _handler_target(request):
    return web.Response(text=TARGET_HTML, content_type="text/html")


@pytest.fixture
async def test_server():
    app = web.Application()
    app.router.add_get("/", _handler_index)
    app.router.add_get("/links", _handler_links)
    app.router.add_get("/form", _handler_form)
    app.router.add_get("/target", _handler_target)
    server = TestServer(app)
    await server.start_server()
    yield server
    await server.close()


@pytest.fixture
async def browser():
    manager = BrowserManager()
    await manager.start()
    browser_tools._manager = manager
    yield manager
    await manager.close()
    browser_tools._manager = None


# ---------------------------------------------------------------------------
# BrowserManager unit tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_browser_manager_starts_with_one_page():
    manager = BrowserManager()
    await manager.start()
    assert manager.page_count == 1
    assert manager.active_index == 0
    assert manager.active_page is not None
    await manager.close()


@pytest.mark.asyncio
async def test_browser_manager_new_page():
    manager = BrowserManager()
    await manager.start()
    await manager.new_page()
    assert manager.page_count == 2
    assert manager.active_index == 1
    await manager.close()


@pytest.mark.asyncio
async def test_browser_manager_switch_to():
    manager = BrowserManager()
    await manager.start()
    await manager.new_page()
    await manager.switch_to(0)
    assert manager.active_index == 0
    await manager.switch_to(1)
    assert manager.active_index == 1
    await manager.close()


@pytest.mark.asyncio
async def test_browser_manager_switch_invalid_index():
    manager = BrowserManager()
    await manager.start()
    with pytest.raises(ToolError):
        await manager.switch_to(5)
    await manager.close()


@pytest.mark.asyncio
async def test_browser_manager_close_page():
    manager = BrowserManager()
    await manager.start()
    await manager.new_page()
    assert manager.page_count == 2
    await manager.close_page(0)
    assert manager.page_count == 1
    assert manager.active_index == 0
    await manager.close()


@pytest.mark.asyncio
async def test_browser_manager_close_last_page_creates_new():
    manager = BrowserManager()
    await manager.start()
    await manager.close_page(0)
    assert manager.page_count == 1
    assert manager.active_index == 0
    await manager.close()


@pytest.mark.asyncio
async def test_ensure_started_idempotent():
    manager = BrowserManager()
    await manager.ensure_started()
    assert manager._browser is not None
    await manager.ensure_started()
    assert manager._browser is not None
    await manager.close()


# ---------------------------------------------------------------------------
# Tool integration tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_browser_open(test_server, browser):
    ctx = MockRunContext()
    url = str(test_server.make_url("/"))
    result = await browser_open(ctx, url)
    assert "Test Page" in result


@pytest.mark.asyncio
async def test_browser_read_page(test_server, browser):
    ctx = MockRunContext()
    url = str(test_server.make_url("/"))
    await browser_open(ctx, url)
    result = await browser_read_page(ctx)
    assert "Hello World" in result
    assert "Test Page" in result


@pytest.mark.asyncio
async def test_browser_find_interactive_elements(test_server, browser):
    ctx = MockRunContext()
    url = str(test_server.make_url("/links"))
    await browser_open(ctx, url)
    result = await browser_find_interactive_elements(ctx)
    assert "First Link" in result
    assert "Button One" in result
    assert "Search" in result


@pytest.mark.asyncio
async def test_browser_click_navigates(test_server, browser):
    ctx = MockRunContext()
    url = str(test_server.make_url("/links"))
    await browser_open(ctx, url)
    result = await browser_click(ctx, "#link1")
    assert "Target Page" in result


@pytest.mark.asyncio
async def test_browser_fill(test_server, browser):
    ctx = MockRunContext()
    url = str(test_server.make_url("/form"))
    await browser_open(ctx, url)
    result = await browser_fill(ctx, "#name", "Karasu")
    assert "Filled" in result


@pytest.mark.asyncio
async def test_browser_type(test_server, browser):
    ctx = MockRunContext()
    url = str(test_server.make_url("/form"))
    await browser_open(ctx, url)
    result = await browser_type(ctx, "#name", "hello")
    assert "Typed" in result


@pytest.mark.asyncio
async def test_browser_select(test_server, browser):
    ctx = MockRunContext()
    url = str(test_server.make_url("/form"))
    await browser_open(ctx, url)
    result = await browser_select(ctx, "#color", "blue")
    assert "Selected" in result


@pytest.mark.asyncio
async def test_browser_press_enter(test_server, browser):
    ctx = MockRunContext()
    url = str(test_server.make_url("/form"))
    await browser_open(ctx, url)
    result = await browser_press(ctx, "Enter", "#name")
    assert "Pressed" in result


@pytest.mark.asyncio
async def test_browser_new_tab(test_server, browser):
    ctx = MockRunContext()
    url = str(test_server.make_url("/"))
    result = await browser_new_tab(ctx, url)
    assert "Opened tab" in result
    assert "Total tabs: 2" in result


@pytest.mark.asyncio
async def test_browser_switch_tab(test_server, browser):
    ctx = MockRunContext()
    url = str(test_server.make_url("/"))
    await browser_new_tab(ctx, url)
    result = await browser_switch_tab(ctx, 0)
    assert "Switched" in result


@pytest.mark.asyncio
async def test_browser_close_tab(test_server, browser):
    ctx = MockRunContext()
    url = str(test_server.make_url("/"))
    await browser_new_tab(ctx, url)
    result = await browser_close_tab(ctx, 1)
    assert "Closed" in result


@pytest.mark.asyncio
async def test_browser_go_back(test_server, browser):
    ctx = MockRunContext()
    url_index = str(test_server.make_url("/"))
    url_links = str(test_server.make_url("/links"))
    await browser_open(ctx, url_index)
    await browser_open(ctx, url_links)
    result = await browser_go_back(ctx)
    assert "Went back" in result


@pytest.mark.asyncio
async def test_browser_go_forward(test_server, browser):
    ctx = MockRunContext()
    url_index = str(test_server.make_url("/"))
    url_links = str(test_server.make_url("/links"))
    await browser_open(ctx, url_index)
    await browser_open(ctx, url_links)
    await browser_go_back(ctx)
    result = await browser_go_forward(ctx)
    assert "Went forward" in result


@pytest.mark.asyncio
async def test_browser_reload(test_server, browser):
    ctx = MockRunContext()
    url = str(test_server.make_url("/"))
    await browser_open(ctx, url)
    result = await browser_reload(ctx)
    assert "Reloaded" in result


@pytest.mark.asyncio
async def test_browser_screenshot(test_server, browser):
    ctx = MockRunContext()
    url = str(test_server.make_url("/"))
    await browser_open(ctx, url)
    result = await browser_screenshot(ctx)
    assert "Screenshot saved" in result


@pytest.mark.asyncio
async def test_browser_wait():
    ctx = MockRunContext()
    result = await browser_wait(ctx, 0.1)
    assert "Waited" in result
