"""自动播放（playback）：按真实经过时间推进的启动/暂停/恢复/查询。

覆盖：
- 状态机：未启动 not_started 查询；启动仅限已开通道且未启动；
  暂停仅限运行；恢复仅限暂停/阻塞；其余 409（detail 带当前 status）；
  未知/已结束会话 404；
- 计时：后台每 50 毫秒以“本段起始游标 + 单调时钟经过整毫秒”推进；
  暂停冻结游标不计时；恢复从保留游标重新开段，不追赶停顿；
- 满额：整批事件容量不足转为 blocked，游标、seq、队列不变；阻塞后
  不自动恢复（确认释放也不恢复、不追赶），须显式 resume；
- 运行时两种手动推进 409；暂停/阻塞下手动推进仍可用；
- 其他既有操作（隐藏切换、接纳、补取、结束）在自动播放期间完整串行；
- 结束停止驱动、按原规则释放容量，之后一切请求 404；旧响应不变。
"""
import threading
import time

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from app import main as main_module  # noqa: E402
from app.main import app  # noqa: E402


def scene_payload(scene_id="scene-auto", limit_note=None):
    # s1[0,1000) s2[800,3000)锁2 s3[1000,2500)：推进到 1000 产生 3 条事件
    return {
        "track_count": 2,
        "expected_revision": 0,
        "subtitles": [
            {"id": "s1", "speaker": "Alice", "start_ms": 0, "end_ms": 1000},
            {"id": "s2", "speaker": "Bob", "start_ms": 800, "end_ms": 3000,
             "locked_track": 2},
            {"id": "s3", "speaker": "Alice", "start_ms": 1000, "end_ms": 2500},
        ],
    }


def make_session(client, scene_id="scene-auto"):
    resp = client.put(f"/scenes/{scene_id}", json=scene_payload())
    assert resp.status_code == 200
    resp = client.post("/sessions", json={"scene_id": scene_id})
    assert resp.status_code == 200
    return resp.json()["session_id"]


def open_channel(client, sid):
    resp = client.post(f"/sessions/{sid}/channel")
    assert resp.status_code == 200
    return resp


def get_status(client, sid):
    resp = client.get(f"/sessions/{sid}/playback")
    assert resp.status_code == 200
    return resp.json()


def wait_for(predicate, timeout=3.0, interval=0.01):
    """轮询至 predicate 为真；超时断言失败（后台节拍 50ms）。"""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(interval)
    raise AssertionError("condition not reached before timeout")


# ---------------------------------------------------------------------------
# 查询与启动前置条件
# ---------------------------------------------------------------------------


def test_status_not_started_before_start():
    client = TestClient(app)
    sid = make_session(client)
    body = get_status(client, sid)
    assert body == {
        "session_id": sid,
        "status": "not_started",
        "cursor_ms": 0,
    }


def test_start_requires_channel_409():
    client = TestClient(app)
    sid = make_session(client)
    resp = client.post(f"/sessions/{sid}/playback/start")
    assert resp.status_code == 409
    assert resp.json()["detail"]["status"] == "not_started"
    # 启动失败不改变状态
    assert get_status(client, sid)["status"] == "not_started"
    # 暂停/恢复对未启动会话也是 409
    assert client.post(f"/sessions/{sid}/playback/pause").status_code == 409
    assert client.post(f"/sessions/{sid}/playback/resume").status_code == 409


def test_start_then_query_running_and_double_start_409():
    client = TestClient(app)
    sid = make_session(client)
    open_channel(client, sid)
    resp = client.post(f"/sessions/{sid}/playback/start")
    assert resp.status_code == 200
    assert resp.json() == {"session_id": sid, "status": "running", "cursor_ms": 0}
    # 重复启动 409，detail 给出当前 running
    resp = client.post(f"/sessions/{sid}/playback/start")
    assert resp.status_code == 409
    assert resp.json()["detail"]["status"] == "running"
    assert get_status(client, sid)["status"] == "running"


def test_playback_endpoints_unknown_session_404():
    client = TestClient(app)
    sid = "nope"
    assert client.get(f"/sessions/{sid}/playback").status_code == 404
    assert client.post(f"/sessions/{sid}/playback/start").status_code == 404
    assert client.post(f"/sessions/{sid}/playback/pause").status_code == 404
    assert client.post(f"/sessions/{sid}/playback/resume").status_code == 404


# ---------------------------------------------------------------------------
# 真实经过时间自动推进；暂停冻结；恢复不追赶
# ---------------------------------------------------------------------------


def test_cursor_advances_with_real_elapsed_time():
    client = TestClient(app)
    sid = make_session(client)
    open_channel(client, sid)
    client.post(f"/sessions/{sid}/playback/start")
    time.sleep(0.23)
    body = get_status(client, sid)
    assert body["status"] == "running"
    # 节拍 50ms：230ms 后游标约为 200ms（允许调度抖动；关键是与挂钟成比例）
    assert 100 <= body["cursor_ms"] <= 400, body


def test_pause_freezes_cursor_and_resume_does_not_catch_up():
    client = TestClient(app)
    sid = make_session(client)
    open_channel(client, sid)
    client.post(f"/sessions/{sid}/playback/start")
    time.sleep(0.16)
    resp = client.post(f"/sessions/{sid}/playback/pause")
    assert resp.status_code == 200
    assert resp.json()["status"] == "paused"
    frozen = resp.json()["cursor_ms"]
    assert frozen >= 100
    # 暂停期间不计时：停顿再久游标都冻结
    time.sleep(0.2)
    assert get_status(client, sid)["cursor_ms"] == frozen

    # 非运行状态再暂停 409；未启动/运行时恢复 409（暂停时可恢复）
    assert client.post(f"/sessions/{sid}/playback/pause").status_code == 409
    assert client.post(f"/sessions/{sid}/playback/start").status_code == 409

    resp = client.post(f"/sessions/{sid}/playback/resume")
    assert resp.status_code == 200
    assert resp.json()["status"] == "running"
    assert resp.json()["cursor_ms"] == frozen
    time.sleep(0.12)
    cur = get_status(client, sid)["cursor_ms"]
    # 恢复后只计恢复之后的经过时间，停顿的 200ms 不被追赶
    assert frozen + 50 <= cur <= frozen + 200, (frozen, cur)


def test_start_after_manual_advance_uses_retained_cursor_as_base():
    # 启动前手动推进过：本段从保留游标（非 0）起计时
    client = TestClient(app)
    sid = make_session(client)
    open_channel(client, sid)
    client.post(f"/sessions/{sid}/advance", json={"time_ms": 900})
    client.post(f"/sessions/{sid}/playback/start")
    time.sleep(0.12)
    cur = get_status(client, sid)["cursor_ms"]
    assert 950 <= cur <= 1050, cur


def test_manual_advance_409_while_running_only():
    client = TestClient(app)
    sid = make_session(client)
    open_channel(client, sid)
    client.post(f"/sessions/{sid}/playback/start")
    # 运行时两种手动推进均 409
    for path in ("advance", "events"):
        resp = client.post(
            f"/sessions/{sid}/{path}", json={"time_ms": 5000}
        )
        assert resp.status_code == 409, (path, resp.status_code)
        assert resp.json()["detail"]["status"] == "running"
    # 暂停后手动推进恢复可用，并成为恢复时的保留游标
    paused_at = client.post(
        f"/sessions/{sid}/playback/pause"
    ).json()["cursor_ms"]
    resp = client.post(
        f"/sessions/{sid}/events", json={"time_ms": paused_at + 400}
    )
    assert resp.status_code == 200
    assert resp.json()["cursor_ms"] == paused_at + 400
    r = client.post(f"/sessions/{sid}/playback/resume")
    assert r.status_code == 200
    assert r.json()["cursor_ms"] == paused_at + 400


# ---------------------------------------------------------------------------
# 满额转阻塞：整批失败、游标/seq/队列不变、显式恢复、不追赶
# ---------------------------------------------------------------------------


def _deterministic_blocked_session(limit=3):
    """构造一个在确定性时刻节拍下阻塞的会话（暂停后手工注入节拍）。

    返回 (client, manager, session, sid)。
    """
    main_module.reset_playback()
    import os
    os.environ["CHANNEL_EVENT_LIMIT"] = str(limit)
    try:
        client = TestClient(app)
        sid = make_session(client)
        open_channel(client, sid)
        manager = main_module.get_playback()
        assert manager.channel_capacity.limit == limit
        session = manager.get(sid)
        client.post(f"/sessions/{sid}/playback/start")
        # 立即暂停，避免真实驱动在注入节拍之间插入推进
        client.post(f"/sessions/{sid}/playback/pause")
        # 手工布置一个从游标 0、时钟原点 0 开始的运行段，逐拍注入
        session.auto_status = "running"
        session.auto_base_cursor_ms = 0
        session.auto_origin_ns = 0
        return client, manager, session, sid
    finally:
        os.environ.pop("CHANNEL_EVENT_LIMIT", None)


def test_capacity_shortage_turns_running_into_blocked_without_partial_effect():
    client, manager, session, sid = _deterministic_blocked_session(limit=3)
    # 节拍到 1000：3 条事件恰好占满容量，推进成功
    manager._auto_tick(session, now_ns=1_000_000_000)
    assert session.auto_status == "running"
    assert session.cursor_ms == 1000
    assert session.channel.next_seq == 4
    assert manager.channel_capacity.used == 3

    # 节拍到 2500：新增 s3 的 exit（第 4 条）整批申请失败 -> blocked
    manager._auto_tick(session, now_ns=2_500_000_000)
    assert session.auto_status == "blocked"
    # 游标、seq、队列保持阻塞前状态
    assert session.cursor_ms == 1000
    assert session.channel.next_seq == 4
    assert [e["seq"] for e in session.channel.events] == [1, 2, 3]
    assert manager.channel_capacity.used == 3
    assert session.auto_origin_ns is None and session.auto_base_cursor_ms is None
    # HTTP 查询同样报 blocked 与保留游标
    body = get_status(client, sid)
    assert body == {"session_id": sid, "status": "blocked", "cursor_ms": 1000}


def test_blocked_session_is_not_driven_and_resume_does_not_catch_up():
    client, manager, session, sid = _deterministic_blocked_session(limit=3)
    manager._auto_tick(session, now_ns=1_000_000_000)
    manager._auto_tick(session, now_ns=2_500_000_000)
    assert session.auto_status == "blocked"
    # 后续节拍（即使时钟已到 10 秒）一律跳过：不推进、不追赶
    manager._auto_tick(session, now_ns=10_000_000_000)
    assert session.auto_status == "blocked" and session.cursor_ms == 1000

    # 确认并删除前缀释放容量——但不自动恢复、不追赶
    client.post(f"/sessions/{sid}/channel/fetch", json={"ack_seq": 0})
    client.post(f"/sessions/{sid}/channel/fetch", json={"ack_seq": 3})
    assert manager.channel_capacity.used == 0
    manager._auto_tick(session, now_ns=10_000_000_000)
    assert session.auto_status == "blocked" and session.cursor_ms == 1000

    # blocked 下暂停/启动 409，恢复 200
    assert client.post(f"/sessions/{sid}/playback/pause").status_code == 409
    resp = client.post(f"/sessions/{sid}/playback/resume")
    assert resp.status_code == 200
    assert resp.json()["cursor_ms"] == 1000
    # 恢复后从 1000 起按真实时间重新计时：再注入一段确定性节拍
    origin = session.auto_origin_ns
    manager._auto_tick(session, now_ns=origin + 1_500_000_000)
    assert session.cursor_ms == 2500
    assert session.channel.next_seq == 5
    last = session.channel.events[-1]
    assert last == {
        "seq": 4, "id": "s3", "speaker": "Alice", "track": 1,
        "time_ms": 2500, "type": "exit",
    }


def test_wall_clock_full_then_fetch_and_resume_via_http(monkeypatch):
    """端到端：小容量下真实驱动自动阻塞，HTTP 确认释放后显式恢复。"""
    monkeypatch.setenv("CHANNEL_EVENT_LIMIT", "3")
    main_module.reset_playback()
    client = TestClient(app)
    sid = make_session(client, "scene-auto-wall")
    open_channel(client, sid)
    client.post(f"/sessions/{sid}/playback/start")

    # 播放越过 1000 毫秒后容量占满，下一节拍（>1000ms）必然阻塞
    wait_for(lambda: get_status(client, sid)["status"] == "blocked", timeout=4)
    body = get_status(client, sid)
    assert body["status"] == "blocked"
    assert body["cursor_ms"] >= 1000
    cursor = body["cursor_ms"]
    # 阻塞期间持续观察游标不变
    time.sleep(0.2)
    assert get_status(client, sid)["cursor_ms"] == cursor

    # 补取一次建立“已返回”上界，再确认全部 3 条释放容量
    fetched = client.post(
        f"/sessions/{sid}/channel/fetch", json={"ack_seq": 0}
    ).json()
    assert len(fetched["events"]) == 3
    resp = client.post(
        f"/sessions/{sid}/channel/fetch", json={"ack_seq": 3}
    )
    assert resp.status_code == 200
    # 容量释放后仍然阻塞，直到显式恢复
    assert get_status(client, sid)["status"] == "blocked"
    resp = client.post(f"/sessions/{sid}/playback/resume")
    assert resp.status_code == 200
    assert resp.json()["status"] == "running"
    time.sleep(0.1)
    assert get_status(client, sid)["status"] == "running"


def test_zero_event_tick_succeeds_when_full():
    """满额时等时（0 事件）节拍不占名额，仍可推进（此处游标不变）。"""
    client, manager, session, sid = _deterministic_blocked_session(limit=3)
    manager._auto_tick(session, now_ns=1_000_000_000)
    assert manager.channel_capacity.remaining == 0
    # 仍在 running：同一时刻（目标==游标，0 事件）的节拍成功，不转阻塞
    manager._auto_tick(session, now_ns=1_000_000_000)
    assert session.auto_status == "running"
    assert session.cursor_ms == 1000


# ---------------------------------------------------------------------------
# 与既有会话操作串行、入队沿用通道规则
# ---------------------------------------------------------------------------


def test_auto_events_enqueued_into_channel_with_window_order_and_seq():
    """自动推进产生的事件沿用窗口/排序/隐藏过滤/seq，可经通道补取。"""
    client = TestClient(app)
    sid = make_session(client, "scene-auto-enq")
    open_channel(client, sid)
    client.post(f"/sessions/{sid}/playback/start")
    wait_for(lambda: get_status(client, sid)["cursor_ms"] >= 3000, timeout=4)
    client.post(f"/sessions/{sid}/playback/pause")
    body = client.post(
        f"/sessions/{sid}/channel/fetch", json={"ack_seq": 0}
    ).json()
    quads = [(e["seq"], e["time_ms"], e["type"], e["id"]) for e in body["events"]]
    assert quads == [
        (1, 800, "enter", "s2"),
        (2, 1000, "exit", "s1"),
        (3, 1000, "enter", "s3"),
        (4, 2500, "exit", "s3"),
        (5, 3000, "exit", "s2"),
    ]


def test_hidden_switch_serialized_with_running_driver():
    """运行中隐藏切换仍可用：显示变更事件入队，其后推进排除隐藏者。"""
    client = TestClient(app)
    sid = make_session(client, "scene-auto-hidden")
    open_channel(client, sid)
    client.post(f"/sessions/{sid}/playback/start")
    # 等游标进入 s1/s2 活动区间（>=800 且 <1000 较难卡，直接在 800~900 附近）
    wait_for(lambda: get_status(client, sid)["cursor_ms"] >= 850, timeout=3)
    client.post(f"/sessions/{sid}/playback/pause")
    frozen = get_status(client, sid)["cursor_ms"]
    # 暂停后做隐藏切换（与推进同锁串行）；若已过 1000，s1 已退出则无其事件
    resp = client.post(
        f"/sessions/{sid}/hidden_speakers",
        json={"hidden_speakers": ["Alice"]},
    )
    assert resp.status_code == 200
    assert all(a["speaker"] != "Alice" for a in resp.json()["active"])
    # 推进与事件窗口此后排除 Alice
    resp = client.post(
        f"/sessions/{sid}/events", json={"time_ms": max(frozen, 1200)}
    )
    assert all(e["speaker"] != "Alice" for e in resp.json()["events"])


def test_accept_and_finish_serialized_with_driver():
    """运行中接纳与结束与后台节拍完整串行；结束后 404 并释放容量。"""
    client = TestClient(app)
    scene_id = "scene-auto-accept"
    sid = make_session(client, scene_id)
    open_channel(client, sid)
    client.post(f"/sessions/{sid}/playback/start")
    time.sleep(0.08)
    # 接纳一个合法修订（延长未开始的 s3）：运行中仍 200
    client.put(f"/scenes/{scene_id}", json={
        "track_count": 2,
        "expected_revision": 1,
        "subtitles": [
            {"id": "s1", "speaker": "Alice", "start_ms": 0, "end_ms": 1000},
            {"id": "s2", "speaker": "Bob", "start_ms": 800, "end_ms": 3000,
             "locked_track": 2},
            {"id": "s3", "speaker": "Alice", "start_ms": 1000, "end_ms": 4000},
        ],
    })
    assert client.post(f"/sessions/{sid}/accept").status_code == 200
    # 结束：一次成功，驱动随即不再推进该会话；其后所有播放控制 404
    resp = client.post(f"/sessions/{sid}/finish")
    assert resp.status_code == 200
    assert client.get(f"/sessions/{sid}/playback").status_code == 404
    for path in (
        "/playback/start",
        "/playback/pause",
        "/playback/resume",
    ):
        assert client.post(f"/sessions/{sid}{path}").status_code == 404
    assert client.post(
        f"/sessions/{sid}/advance", json={"time_ms": 99999}
    ).status_code == 404
    assert client.post(
        f"/sessions/{sid}/events", json={"time_ms": 99999}
    ).status_code == 404


def test_concurrent_pause_and_tick_no_progress_after_response():
    """暂停响应返回后，旧节拍不得再推进（暂停与节拍在会话锁内串行）。"""
    client, manager, session, sid = _deterministic_blocked_session(limit=1024)
    # 恢复真实驱动（去掉注入态）：以当前挂钟重新开段
    client.post(f"/sessions/{sid}/playback/resume")
    time.sleep(0.1)
    client.post(f"/sessions/{sid}/playback/pause")
    frozen = session.cursor_ms
    # 并发制造大量“旧时刻”节拍与查询：暂停后一律不得推进
    errors = []

    def hammer():
        for _ in range(200):
            try:
                manager._auto_tick(session, now_ns=10**18)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

    threads = [threading.Thread(target=hammer) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert session.auto_status == "paused"
    assert session.cursor_ms == frozen


def test_old_responses_unchanged():
    """创建/推进/通道等旧响应字段不因自动播放功能而变化。"""
    client = TestClient(app)
    client.put("/scenes/old-resp", json=scene_payload())
    body = client.post(
        "/sessions", json={"scene_id": "old-resp"}
    ).json()
    assert set(body) == {
        "session_id", "scene_id", "source_revision", "cursor_ms",
        "track_count", "changes", "subtitles", "assignments",
    }
    sid = body["session_id"]
    adv = client.post(f"/sessions/{sid}/advance", json={"time_ms": 900}).json()
    assert set(adv) == {
        "session_id", "scene_id", "source_revision", "cursor_ms", "active",
    }
    open_channel(client, sid)
    assert client.post(f"/sessions/{sid}/channel").status_code == 409
    client.post(f"/sessions/{sid}/finish")
    assert client.get(f"/sessions/{sid}/playback").status_code == 404
