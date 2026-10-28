"""外部献血业务接口契约；Agent 只依赖此协议与传输适配器。"""

from __future__ import annotations

import hashlib
from typing import Protocol


class LegacyBloodAPI(Protocol):
    """公司既有献血业务 HTTP API 的稳定领域契约。"""

    transport_name: str

    def list_rewards(self) -> list[dict]:
        """查询当前可展示的礼品目录。"""
        ...

    def redeem_reward(self, user_id: str, reward_id: str, idempotency_key: str,
                       address_id: str | None = None) -> dict:
        """按用户、礼品和收货地址提交积分兑换。"""
        ...

    def get_user_profile(self, user_id: str) -> dict:
        """查询指定用户的实名资料。"""
        ...

    def get_points(self, user_id: str) -> dict:
        """查询指定用户的当前积分。"""
        ...

    def get_donation_records(self, user_id: str) -> dict:
        """查询指定用户的献血记录。"""
        ...

    def get_redemption_orders(self, user_id: str) -> dict:
        """查询指定用户的积分兑换订单。"""
        ...

    def cancel_redemption_order(self, user_id: str, redemption_id: str) -> dict:
        """取消用户指定的可取消兑换订单。"""
        ...

    def get_shipping_addresses(self, user_id: str) -> dict:
        """查询指定用户的收货地址簿。"""
        ...

    def save_shipping_address(self, user_id: str, recipient: str | None, mobile: str | None,
                              detail: str | None, address_id: str | None = None) -> dict:
        """新增收货地址，或按地址编号修改已有地址。"""
        ...

    def set_default_shipping_address(self, user_id: str, address_id: str) -> dict:
        """设置指定用户的默认收货地址。"""
        ...

    def set_order_shipping_address(self, user_id: str, redemption_id: str, address_id: str) -> dict:
        """为指定兑换订单关联已有收货地址。"""
        ...

    def get_appointment_history(self, user_id: str) -> list[dict]:
        """查询用户历史预约偏好，供新预约参考。"""
        ...

    def get_user_appointments(self, user_id: str) -> dict:
        """查询用户已提交预约及其当前状态。"""
        ...

    def get_sites_and_slots(self, user_id: str) -> list[dict]:
        """查询当前可预约网点及日期时段。"""
        ...

    def submit_appointment(self, payload: dict, idempotency_key: str) -> dict:
        """提交已确认的预约并返回接口回执。"""
        ...

    def cancel_appointment(self, user_id: str, appointment_id: str) -> dict:
        """取消用户指定的已提交预约。"""
        ...

    def checkin_appointment(self, user_id: str, appointment_id: str) -> dict:
        """为用户指定的预约执行签到。"""
        ...


def appointment_idempotency_key(user_id: str, slots: dict) -> str:
    """由预约请求生成稳定的幂等键。"""
    stable = (f"{user_id}:{slots.get('site')}:{slots.get('date')}:{slots.get('time_slot')}:"
              f"{slots.get('donor_kind')}:{slots.get('donor_name')}:{slots.get('donor_mobile')}:"
              f"{slots.get('donor_id_card')}:{slots.get('donation_type')}:{slots.get('donation_amount')}")
    return hashlib.sha256(stable.encode()).hexdigest()
