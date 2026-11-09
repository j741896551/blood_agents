"""外部献血业务 HTTP 客户端；不含服务端路由和业务数据库逻辑。"""

from __future__ import annotations

from typing import Any
from urllib.parse import quote

import httpx


class LegacyAPITransportError(RuntimeError):
    """HTTP 服务不可达或返回了非成功 HTTP 状态；不能把它误当成空业务结果。"""


class LegacyAPIContractError(RuntimeError):
    """服务返回了与 Agent 侧接口契约不一致的数据。"""


class LegacyHTTPClient:
    """HTTP adapter：方法名和返回契约对齐 LegacyBloodAPI。

    `http_client` 可注入测试用传输层，便于验证协议边界。
    查询失败会抛出可识别异常，绝不伪装成“无记录”或“没有可预约时段”。
    """

    transport_name = "legacy_http"

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 10.0,
        http_client: httpx.Client | None = None,
    ) -> None:
        """初始化 LegacyHTTPClient 的依赖与运行状态。"""
        if not base_url.strip():
            raise ValueError("legacy HTTP base_url 不能为空")
        self._owns_client = http_client is None
        self._client = http_client or httpx.Client(
            base_url=base_url.rstrip("/"), timeout=timeout,
        )

    def close(self) -> None:
        """仅关闭由本 adapter 创建的连接池。"""
        if self._owns_client:
            self._client.close()

    @staticmethod
    def _segment(value: str) -> str:
        """把入口 ID 编码成单一路径段，避免特殊字符改变路由结构。"""
        return quote(value, safe="")

    def _get(self, path: str) -> Any:
        """执行 GET 请求并解析业务响应。"""
        try:
            response = self._client.get(path)
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, ValueError) as exc:
            # 不把 URL、完整响应体或用户资料写入异常，减少意外泄漏风险。
            raise LegacyAPITransportError(
                f"外部业务 HTTP 查询失败（{type(exc).__name__}）"
            ) from exc

    def _post(self, path: str, payload: dict[str, Any]) -> Any:
        """发送 POST 请求并解析业务响应。"""
        try:
            response = self._client.post(path, json=payload)
            if response.status_code == 422:
                detail = response.json().get("detail", "业务请求未通过校验")
                raise ValueError(str(detail))
            response.raise_for_status()
            return response.json()
        except ValueError:
            raise
        except (httpx.HTTPError, TypeError) as exc:
            raise LegacyAPITransportError(
                f"外部业务 HTTP 请求失败（{type(exc).__name__}）"
            ) from exc

    def _dict(self, path: str) -> dict[str, Any]:
        """发送请求并把响应解析为字典。"""
        data = self._get(path)
        if not isinstance(data, dict):
            raise LegacyAPIContractError("旧业务 HTTP 接口应返回 JSON object")
        return data

    def _post_dict(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        """发送 POST 请求并把响应解析为字典。"""
        data = self._post(path, payload)
        if not isinstance(data, dict):
            raise LegacyAPIContractError("旧业务 HTTP 接口应返回 JSON object")
        return data

    def _list(self, path: str) -> list[dict[str, Any]]:
        """把接口响应解析为列表。"""
        data = self._get(path)
        if not isinstance(data, list) or any(not isinstance(item, dict) for item in data):
            raise LegacyAPIContractError("旧业务 HTTP 接口应返回 JSON object 列表")
        return data

    def list_rewards(self) -> list[dict[str, Any]]:
        """列出当前可供查询的礼品目录。"""
        return self._list("/v1/rewards")

    def get_user_profile(self, user_id: str) -> dict[str, Any]:
        """按用户查询旧业务系统中的资料。"""
        return self._dict(f"/v1/users/{self._segment(user_id)}/profile")

    def get_points(self, user_id: str) -> dict[str, Any]:
        """按用户查询当前积分余额。"""
        return self._dict(f"/v1/users/{self._segment(user_id)}/points")

    def get_donation_records(self, user_id: str) -> dict[str, Any]:
        """按用户查询已有献血记录。"""
        return self._dict(f"/v1/users/{self._segment(user_id)}/donation-records")

    def get_redemption_orders(self, user_id: str) -> dict[str, Any]:
        """按用户查询积分兑换订单列表。"""
        return self._dict(f"/v1/users/{self._segment(user_id)}/redemption-orders")

    def get_shipping_addresses(self, user_id: str) -> dict[str, Any]:
        """按用户查询可用的收货地址。"""
        return self._dict(f"/v1/users/{self._segment(user_id)}/shipping-addresses")

    def get_appointment_history(self, user_id: str) -> list[dict[str, Any]]:
        """按用户查询用于预约推荐的历史偏好。"""
        return self._list(f"/v1/users/{self._segment(user_id)}/appointment-history")

    def get_user_appointments(self, user_id: str) -> dict[str, Any]:
        """按用户查询已提交预约及状态。"""
        return self._dict(f"/v1/users/{self._segment(user_id)}/appointments")

    def get_sites_and_slots(self, user_id: str) -> list[dict[str, Any]]:
        """查询当前网点及其可预约时段。"""
        return self._list(f"/v1/users/{self._segment(user_id)}/sites-and-slots")

    def submit_appointment(self, payload: dict, idempotency_key: str) -> dict[str, Any]:
        """提交预约并返回接口确认或拒绝原因。"""
        return self._post_dict("/v1/appointments", {
            "payload": payload, "idempotency_key": idempotency_key,
        })

    def cancel_appointment(self, user_id: str, appointment_id: str) -> dict[str, Any]:
        """取消指定预约并返回业务接口回执。"""
        return self._post_dict(
            f"/v1/users/{self._segment(user_id)}/appointments/{self._segment(appointment_id)}/cancel",
            {},
        )

    def checkin_appointment(self, user_id: str, appointment_id: str) -> dict[str, Any]:
        """对指定预约执行签到并返回业务接口回执。"""
        return self._post_dict(
            f"/v1/users/{self._segment(user_id)}/appointments/{self._segment(appointment_id)}/checkin",
            {},
        )

    def redeem_reward(self, user_id: str, reward_id: str, idempotency_key: str,
                      address_id: str | None = None) -> dict[str, Any]:
        """提交礼品兑换并返回积分与订单回执。"""
        return self._post_dict(f"/v1/users/{self._segment(user_id)}/redemptions", {
            "reward_id": reward_id, "idempotency_key": idempotency_key,
            "address_id": address_id,
        })

    def cancel_redemption_order(self, user_id: str, redemption_id: str) -> dict[str, Any]:
        """取消指定兑换订单并返回业务接口回执。"""
        return self._post_dict(
            f"/v1/users/{self._segment(user_id)}/redemptions/{self._segment(redemption_id)}/cancel",
            {},
        )

    def save_shipping_address(self, user_id: str, recipient: str | None, mobile: str | None,
                              detail: str | None, address_id: str | None = None) -> dict[str, Any]:
        """新增或修改用户的收货地址。"""
        return self._post_dict(f"/v1/users/{self._segment(user_id)}/shipping-addresses", {
            "recipient": recipient, "mobile": mobile, "detail": detail,
            "address_id": address_id,
        })

    def set_default_shipping_address(self, user_id: str, address_id: str) -> dict[str, Any]:
        """设置用户指定的默认收货地址。"""
        return self._post_dict(
            f"/v1/users/{self._segment(user_id)}/shipping-addresses/default", {
            "address_id": address_id,
        })

    def set_order_shipping_address(self, user_id: str, redemption_id: str,
                                   address_id: str) -> dict[str, Any]:
        """为兑换订单设置收货地址。"""
        return self._post_dict(
            f"/v1/users/{self._segment(user_id)}/redemptions/"
            f"{self._segment(redemption_id)}/shipping-address",
            {"address_id": address_id},
        )


