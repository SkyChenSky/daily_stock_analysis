# -*- coding: utf-8 -*-
"""
===================================
BaiduFetcher - 百度财经数据源
===================================

数据来源：百度财经 API (finance.pae.baidu.com)
特点：免费、无需 Token、接口简洁

API 说明：
- 日K线接口：quotation_kline_ab
- 分钟线接口：quotation_minute_ab
- 返回格式：JSON，其中 marketData 为 CSV 格式字符串
  （分号分隔行，逗号分隔字段）

防封禁策略：
1. 需先访问 gushitong.baidu.com 获取 Cookie（BAIDUID）
2. 每次请求前随机休眠
3. 随机 User-Agent 轮换
4. 请求超时保护
"""

import logging
import os
import random
import time
from typing import Optional, Dict, Any, List, Tuple

import pandas as pd
import requests

try:
    from .base import (
        BaseFetcher, DataFetchError, STANDARD_COLUMNS,
        _is_hk_market, _is_etf_code, normalize_stock_code,
    )
    from .realtime_types import (
        UnifiedRealtimeQuote, RealtimeSource,
        safe_float, safe_int, get_realtime_circuit_breaker,
    )
except ImportError:
    # 支持直接运行: python data_provider/baidu_fetcher.py
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from data_provider.base import (
        BaseFetcher, DataFetchError, STANDARD_COLUMNS,
        _is_hk_market, _is_etf_code, normalize_stock_code,
    )
    from data_provider.realtime_types import (
        UnifiedRealtimeQuote, RealtimeSource,
        safe_float, safe_int, get_realtime_circuit_breaker,
    )

logger = logging.getLogger(__name__)

# 百度财经 API 基础 URL
BAIDU_FINANCE_API_URL = "https://finance.pae.baidu.com/vapi/v1/getquotation"

# 股票通页面（用于获取 Cookie）
GUSHITONG_BASE_URL = "https://gushitong.baidu.com/stock/ab-{code}"

# User-Agent 池
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15',
]

# 默认请求超时（秒）
_DEFAULT_TIMEOUT = 15


def _safe_float(val: str) -> float:
    """安全转换字符串为浮点数，处理空值和 '--' 占位符及 '+' 前缀"""
    val = val.strip()
    if not val or val == '--':
        return 0.0
    return float(val)


def _is_block_code(code: str) -> bool:
    """判断是否为百度概念/行业板块代码（如 220000 基础化工）"""
    normalized = normalize_stock_code(code)
    return normalized.isdigit() and len(normalized) == 6 and normalized.startswith('22')


class BaiduFetcher(BaseFetcher):
    """
    百度财经数据源实现

    优先级：1（与 AkshareFetcher 同级）
    数据来源：百度财经 API
    接口：finance.pae.baidu.com

    API 返回的 marketData 格式（日K线，每行 19 个字段）：
    字段顺序：timestamp, time, open, close, volume, high, low, amount,
              range(涨跌额), ratio(涨跌幅), turnoverratio, preClose,
              ma5avgprice, ma5volume, ma10avgprice, ma10volume,
              ma20avgprice, ma20volume, (可能有第19字段)

    注意：需要先访问 gushitong.baidu.com 获取 BAIDUID Cookie，
    否则 API 会返回 403 Forbidden。
    """

    name = "BaiduFetcher"
    priority = int(os.getenv("BAIDU_PRIORITY", "1"))

    def __init__(self, sleep_min: float = 0.5, sleep_max: float = 1.5):
        """
        初始化 BaiduFetcher

        Args:
            sleep_min: 最小休眠时间（秒）
            sleep_max: 最大休眠时间（秒）
        """
        self.sleep_min = sleep_min
        self.sleep_max = sleep_max
        self._last_request_time: Optional[float] = None
        self._session = requests.Session()
        self._session.headers.update({
            'User-Agent': random.choice(USER_AGENTS),
        })
        self._session_cookies_ready = False

    def _ensure_cookies(self, stock_code: str, referer_path: str = None) -> None:
        """
        确保已获取百度 Cookie（BAIDUID）

        访问 gushitong.baidu.com 的股票页面来获取必要的 Cookie，
        之后才能成功调用 K 线数据 API。
        BAIDUID 是域级别 Cookie，即使页面返回非 200 也可能通过重定向设置。
        """
        if self._session_cookies_ready:
            return
        try:
            if referer_path:
                page_url = f"https://gushitong.baidu.com/{referer_path}"
            else:
                normalized = normalize_stock_code(stock_code)
                if _is_hk_market(normalized):
                    api_code = normalized[2:] if normalized.upper().startswith('HK') else normalized
                    page_url = f"https://gushitong.baidu.com/stock/hk-{api_code}"
                else:
                    page_url = GUSHITONG_BASE_URL.format(code=stock_code)
            self._session.get(page_url, timeout=_DEFAULT_TIMEOUT)
            if 'BAIDUID' in dict(self._session.cookies):
                self._session_cookies_ready = True
                logger.debug("[BaiduFetcher] Cookie 获取成功")
            else:
                logger.warning("[BaiduFetcher] Cookie 获取异常: BAIDUID 未设置")
        except requests.exceptions.RequestException as e:
            logger.warning(f"[BaiduFetcher] Cookie 获取失败: {e}")

    def _enforce_rate_limit(self) -> None:
        """速率限制：随机休眠"""
        if self._last_request_time is not None:
            elapsed = time.time() - self._last_request_time
            min_interval = random.uniform(self.sleep_min, self.sleep_max)
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
        self._last_request_time = time.time()

    def _get_api_config(self, stock_code: str) -> Dict[str, str]:
        """
        根据股票代码判断类型，返回对应的百度 API 参数

        Returns:
            dict with keys: group, market_type, code, is_kc, extra_params, referer_path
        """
        normalized = normalize_stock_code(stock_code)

        # 港股：5位数字代码（如 00700、01801）
        if _is_hk_market(normalized):
            api_code = normalized[2:] if normalized.upper().startswith('HK') else normalized
            return {
                'group': 'quotation_kline_hk',
                'market_type': 'hk',
                'code': api_code,
                'is_kc': '0',
                'extra_params': f'&query={api_code}',
                'referer_path': f'stock/hk-{api_code}',
            }

        # ETF/基金（如 512400、159919）
        if _is_etf_code(normalized):
            return {
                'group': 'quotation_kline_ab',
                'market_type': 'ab',
                'code': normalized,
                'is_kc': '0',
                'extra_params': f'&query={normalized}&financeType=etf',
                'referer_path': f'stock/ab-{normalized}',
            }

        # 概念/行业板块（如 220000 基础化工）
        if _is_block_code(normalized):
            return {
                'group': 'quotation_block_kline',
                'market_type': 'ab',
                'code': normalized,
                'is_kc': '0',
                'extra_params': f'&query={normalized}',
                'referer_path': f'stock/ab-{normalized}',
            }

        # 普通A股
        is_kc = "1" if normalized.startswith('3') or normalized.startswith('688') else "0"
        return {
            'group': 'quotation_kline_ab',
            'market_type': 'ab',
            'code': normalized,
            'is_kc': is_kc,
            'extra_params': '',
            'referer_path': f'stock/ab-{normalized}',
        }

    def _get_realtime_api_config(self, stock_code: str, stock_type: str = None) -> Dict[str, str]:
        """
        根据股票代码判断类型，返回对应的百度实时行情 API 参数

        Args:
            stock_code: 股票代码（已标准化）
            stock_type: 可选类型提示，"block"=行业板块，"concept"=概念板块

        Returns:
            dict with keys: group, market_type, code, is_kc, extra_params,
                            referer_path, use_srcid, double_fin_client
        """
        normalized = normalize_stock_code(stock_code)

        # 港股
        if _is_hk_market(normalized):
            api_code = normalized[2:] if normalized.upper().startswith('HK') else normalized
            return {
                'group': 'quotation_minute_hk',
                'market_type': 'hk',
                'code': api_code,
                'is_kc': '0',
                'extra_params': '',
                'referer_path': f'stock/hk-{api_code}',
                'use_srcid': True,
                'double_fin_client': True,
                'include_is_kc': True,
            }

        # 行业/概念板块（显式指定或代码以22开头）
        if stock_type in ('block', 'concept') or _is_block_code(normalized):
            return {
                'group': 'quotation_block_minute',
                'market_type': 'ab',
                'code': normalized,
                'is_kc': '',
                'extra_params': '',
                'referer_path': f'stock/ab-{normalized}',
                'use_srcid': False,
                'double_fin_client': True,
                'include_is_kc': False,
            }

        # ETF/基金
        if _is_etf_code(normalized):
            return {
                'group': 'quotation_minute_ab',
                'market_type': 'ab',
                'code': normalized,
                'is_kc': '0',
                'extra_params': '&financeType=etf',
                'referer_path': f'stock/ab-{normalized}',
                'use_srcid': True,
                'double_fin_client': True,
                'include_is_kc': True,
            }

        # 普通A股
        is_kc = "1" if normalized.startswith('3') or normalized.startswith('688') else "0"
        return {
            'group': 'quotation_minute_ab',
            'market_type': 'ab',
            'code': normalized,
            'is_kc': is_kc,
            'extra_params': '',
            'referer_path': f'stock/ab-{normalized}',
            'use_srcid': True,
            'double_fin_client': True,
            'include_is_kc': True,
        }

    @staticmethod
    def _parse_pankou_value(pankou_list: list, ename: str) -> Optional[float]:
        """
        从 pankouinfos.list 中提取指定 ename 的 originValue 并转为 float

        Args:
            pankou_list: pankouinfos.list 数组
            ename: 指标英文名 (如 'open', 'volumeRatio')

        Returns:
            浮点数值，未找到或转换失败返回 None
        """
        for item in pankou_list:
            if item.get('ename') == ename:
                origin = item.get('originValue')
                if origin is not None:
                    return _safe_float(str(origin))
                return None
        return None

    def get_realtime_quote(self, stock_code: str, stock_type: str = None, stock_name: str = None) -> Optional[UnifiedRealtimeQuote]:
        """
        从百度财经获取实时行情数据

        支持类型：A股、港股、ETF/基金、行业板块、概念板块

        Args:
            stock_code: 股票代码
            stock_type: 可选类型提示，"block"=行业板块，"concept"=概念板块

        Returns:
            UnifiedRealtimeQuote 对象，失败返回 None
        """
        normalized = normalize_stock_code(stock_code)

        # 检查熔断器
        cb = get_realtime_circuit_breaker()
        if not cb.is_available("baidu"):
            logger.debug("[BaiduFetcher] 实时行情源 baidu 处于熔断状态，跳过")
            return None

        config = self._get_realtime_api_config(normalized, stock_type=stock_type)

        # 确保 Cookie
        self._ensure_cookies(normalized, config['referer_path'])

        # 速率限制
        self._enforce_rate_limit()

        logger.info(
            f"[BaiduFetcher] 请求实时行情: stock_code={normalized}, "
            f"type={config['group']}"
        )

        # 构建请求 URL
        url_parts = [f"{BAIDU_FINANCE_API_URL}?"]
        if config['use_srcid']:
            url_parts.append("srcid=5353&")
        url_parts.append(f"pointType=string&group={config['group']}")
        url_parts.append(f"&query={config['code']}")
        url_parts.append(f"&code={config['code']}&market_type={config['market_type']}")
        url_parts.append("&newFormat=1")
        if stock_name:
            from urllib.parse import quote
            url_parts.append(f"&name={quote(stock_name)}")
        if config.get('include_is_kc', True):
            url_parts.append(f"&is_kc={config['is_kc']}")
        url_parts.append(config.get('extra_params', ''))
        url_parts.append("&finClientType=pc")
        if config['double_fin_client']:
            url_parts.append("&finClientType=pc")
        url = ''.join(url_parts)

        headers = {
            'Referer': f"https://gushitong.baidu.com/{config['referer_path']}",
        }

        try:
            response = self._session.get(
                url,
                headers=headers,
                timeout=_DEFAULT_TIMEOUT,
            )
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException as e:
            if hasattr(e, 'response') and e.response is not None and e.response.status_code == 403:
                self._session_cookies_ready = False
            cb.record_failure("baidu", str(e))
            logger.warning(f"[BaiduFetcher] 实时行情请求失败: {e}")
            return None
        except ValueError as e:
            cb.record_failure("baidu", str(e))
            logger.warning(f"[BaiduFetcher] 实时行情 JSON 解析失败: {e}")
            return None

        try:
            result = data.get('Result', {})
            if not result:
                cb.record_failure("baidu", "Result 为空")
                return None

            # 提取 pankouinfos
            pankou_list = result.get('pankouinfos', {}).get('list', [])
            basic = result.get('basicinfos', {})
            cur = result.get('cur', {})

            code = basic.get('code', normalized)
            name = basic.get('name', '')

            # 从 cur 提取最新价、涨跌幅、涨跌额
            raw_price = cur.get('price')
            price = safe_float(raw_price) if raw_price else None

            raw_ratio = cur.get('ratio', '')
            change_pct = safe_float(str(raw_ratio).replace('%', ''))

            raw_increase = cur.get('increase', '')
            change_amount = safe_float(str(raw_increase))

            # 从 pankouinfos 提取各项指标
            quote = UnifiedRealtimeQuote(
                code=code,
                name=name,
                source=RealtimeSource.BAIDU,
                price=price,
                change_pct=change_pct,
                change_amount=change_amount,
                volume=safe_int(self._parse_pankou_value(pankou_list, 'volume')),
                amount=safe_float(self._parse_pankou_value(pankou_list, 'amount')),
                volume_ratio=self._parse_pankou_value(pankou_list, 'volumeRatio'),
                turnover_rate=self._parse_pankou_value(pankou_list, 'turnoverRatio'),
                amplitude=self._parse_pankou_value(pankou_list, 'amplitudeRatio'),
                open_price=self._parse_pankou_value(pankou_list, 'open'),
                high=self._parse_pankou_value(pankou_list, 'high'),
                low=self._parse_pankou_value(pankou_list, 'low'),
                pre_close=self._parse_pankou_value(pankou_list, 'preClose'),
                pe_ratio=self._parse_pankou_value(pankou_list, 'peratio'),
                pb_ratio=self._parse_pankou_value(pankou_list, 'bvRatio'),
                total_mv=safe_float(self._parse_pankou_value(pankou_list, 'capitalization')),
                circ_mv=safe_float(self._parse_pankou_value(pankou_list, 'currencyValue')),
                high_52w=self._parse_pankou_value(pankou_list, 'w52_high'),
                low_52w=self._parse_pankou_value(pankou_list, 'w52_low'),
            )

            if not quote.has_basic_data():
                cb.record_failure("baidu", "price 无有效数据")
                logger.debug(f"[BaiduFetcher] {normalized} 实时行情 price 无效: {price}")
                return None

            cb.record_success("baidu")
            logger.info(f"[BaiduFetcher] {normalized} 实时行情获取成功: {name} {price}")
            return quote

        except Exception as e:
            cb.record_failure("baidu", str(e))
            logger.warning(f"[BaiduFetcher] 实时行情解析异常: {e}")
            return None

    def _fetch_raw_data(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        从百度财经获取原始 K 线数据

        支持类型：A股、港股、ETF/基金、概念/行业板块

        Args:
            stock_code: 股票代码
            start_date: 开始日期 'YYYY-MM-DD'
            end_date: 结束日期 'YYYY-MM-DD'

        Returns:
            原始 DataFrame
        """
        # 根据代码类型获取 API 参数
        config = self._get_api_config(stock_code)

        # 确保有 Cookie
        self._ensure_cookies(stock_code, config['referer_path'])

        # 速率限制
        self._enforce_rate_limit()

        logger.info(
            f"[BaiduFetcher] 请求 K 线数据: stock_code={stock_code}, "
            f"type={config['group']}, range={start_date}~{end_date}"
        )

        # 百度 API 要求 finClientType=pc 出现两次，手动拼接 URL
        url = (
            f"{BAIDU_FINANCE_API_URL}?"
            f"srcid=5353&pointType=string&group={config['group']}"
            f"&code={config['code']}&market_type={config['market_type']}"
            f"&newFormat=1&is_kc={config['is_kc']}&ktype=day"
            f"{config['extra_params']}"
            f"&finClientType=pc&finClientType=pc"
        )

        headers = {
            'Referer': f"https://gushitong.baidu.com/{config['referer_path']}",
        }

        try:
            response = self._session.get(
                url,
                headers=headers,
                timeout=_DEFAULT_TIMEOUT,
            )
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException as e:
            # 如果是 403，可能是 Cookie 过期，重置以便下次重新获取
            if hasattr(e, 'response') and e.response is not None and e.response.status_code == 403:
                self._session_cookies_ready = False
            raise DataFetchError(f"[BaiduFetcher] 请求失败: {e}") from e
        except ValueError as e:
            raise DataFetchError(f"[BaiduFetcher] 响应 JSON 解析失败: {e}") from e

        # 解析 marketData
        try:
            result = data.get('Result', {})
            new_market_data = result.get('newMarketData', {})
            market_data_str = new_market_data.get('marketData', '')

            if not market_data_str:
                raise DataFetchError(f"[BaiduFetcher] {stock_code} 未返回 marketData")
        except (AttributeError, TypeError) as e:
            raise DataFetchError(f"[BaiduFetcher] {stock_code} 响应结构异常: {e}") from e

        # 解析 CSV 格式数据
        rows = self._parse_market_data(market_data_str)
        if not rows:
            raise DataFetchError(f"[BaiduFetcher] {stock_code} 解析后无有效数据")

        df = pd.DataFrame(rows)

        # 日期过滤
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            df = df[(df['date'] >= start_dt) & (df['date'] <= end_dt)]

        logger.info(f"[BaiduFetcher] {stock_code} 获取原始数据: {len(df)} 条")
        return df

    def _parse_market_data(self, market_data_str: str) -> List[Dict[str, Any]]:
        """
        解析百度财经 marketData CSV 字符串

        格式：分号分隔行，逗号分隔字段
        字段顺序（19个）：
        0: timestamp (秒级时间戳)
        1: time (日期字符串, 如 "2024-01-15")
        2: open (开盘价)
        3: close (收盘价)
        4: volume (成交量)
        5: high (最高价)
        6: low (最低价)
        7: amount (成交额)
        8: range (涨跌额, 带正负号如 "+0.10")
        9: ratio (涨跌幅, 带正负号如 "+0.95")
        10: turnoverratio (换手率)
        11: preClose (前收盘价)
        12-17: ma5/ma10/ma20 均价和成交量 (可能为 "--")

        日K线数据中日期字段不含空格（格式 "YYYY-MM-DD"），
        分钟线数据日期含空格（格式 "YYYY-MM-DD HH:MM"），
        通过空格过滤只保留日K线数据。

        Args:
            market_data_str: CSV 格式字符串

        Returns:
            解析后的字典列表
        """
        rows = []
        seen_dates = set()

        line_items = market_data_str.split(';')
        for item in line_items:
            item = item.strip()
            if not item:
                continue

            fields = item.split(',')
            if len(fields) < 10:
                continue

            try:
                # 提取日期（字段1）
                date_str = fields[1].strip()
                # 过滤：只要纯日期格式（不含空格，即日线数据）
                if ' ' in date_str:
                    continue

                # 标准化日期格式
                if '-' in date_str:
                    normalized_date = date_str
                elif len(date_str) == 8:
                    normalized_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
                else:
                    continue

                # 去重：同一日期只保留第一条
                if normalized_date in seen_dates:
                    continue
                seen_dates.add(normalized_date)

                # 百度返回的涨跌幅可能带 "+" 前缀，如 "+0.95"
                row = {
                    'date': normalized_date,
                    'open': _safe_float(fields[2]),
                    'close': _safe_float(fields[3]),
                    'volume': _safe_float(fields[4]),
                    'high': _safe_float(fields[5]),
                    'low': _safe_float(fields[6]),
                    'amount': _safe_float(fields[7]),
                    'pct_chg': _safe_float(fields[9]),
                }
                rows.append(row)
            except (ValueError, IndexError) as e:
                logger.debug(f"[BaiduFetcher] 跳过异常数据行: {item[:50]}... ({e})")
                continue

        return rows

    def _normalize_data(self, df: pd.DataFrame, stock_code: str) -> pd.DataFrame:
        """
        标准化百度财经数据

        _parse_market_data 已将字段映射为标准列名，
        此处只需确保列顺序和完整性。

        标准列名：date, open, high, low, close, volume, amount, pct_chg
        """
        df = df.copy()

        # 确保所有标准列都存在
        for col in STANDARD_COLUMNS:
            if col not in df.columns:
                df[col] = 0

        # 只保留标准列
        df = df[STANDARD_COLUMNS]

        return df

if __name__ == "__main__":

    # 测试代码
    logging.basicConfig(level=logging.INFO)

    fetcher = BaiduFetcher()

    # 测试 A 股实时行情
    print("=" * 50)
    print("测试 A 股实时行情获取")
    print("=" * 50)
    try:
        quote = fetcher.get_realtime_quote('002497')  # 贵州茅台
        if quote:
            print(f"[A股实时] {quote.name}: 价格={quote.price}, 涨跌幅={quote.change_pct}%")
            print(f"  开盘={quote.open_price}, 最高={quote.high}, 最低={quote.low}, 昨收={quote.pre_close}")
            print(f"  成交量={quote.volume}, 成交额={quote.amount}")
            print(f"  量比={quote.volume_ratio}, 换手率={quote.turnover_rate}, 振幅={quote.amplitude}")
            print(f"  PE={quote.pe_ratio}, PB={quote.pb_ratio}")
            print(f"  总市值={quote.total_mv}, 流通市值={quote.circ_mv}")
            print(f"  52周高={quote.high_52w}, 52周低={quote.low_52w}")
            print(f"  source={quote.source}")
        else:
            print("[A股实时] 未获取到数据")
    except Exception as e:
        print(f"[A股实时] 获取失败: {e}")

    # 测试 ETF 实时行情
    print("\n" + "=" * 50)
    print("测试 ETF 实时行情获取")
    print("=" * 50)
    try:
        quote = fetcher.get_realtime_quote('513310')  # 证券ETF
        if quote:
            print(f"[ETF实时] {quote.name}: 价格={quote.price}, 涨跌幅={quote.change_pct}%")
        else:
            print("[ETF实时] 未获取到数据")
    except Exception as e:
        print(f"[ETF实时] 获取失败: {e}")

    # 测试港股实时行情
    print("\n" + "=" * 50)
    print("测试港股实时行情获取")
    print("=" * 50)
    try:
        quote = fetcher.get_realtime_quote('00700')  # 腾讯控股
        if quote:
            print(f"[港股实时] {quote.name}: 价格={quote.price}, 涨跌幅={quote.change_pct}%")
        else:
            print("[港股实时] 未获取到数据")
    except Exception as e:
        print(f"[港股实时] 获取失败: {e}")

    # 测试行业板块实时行情
    print("\n" + "=" * 50)
    print("测试行业板块实时行情获取")
    print("=" * 50)
    try:
        quote = fetcher.get_realtime_quote('220000', stock_name='基础化工')
        if quote:
            print(f"[行业板块实时] {quote.name}: 价格={quote.price}, 涨跌幅={quote.change_pct}%")
        else:
            print("[行业板块实时] 未获取到数据（可能需要特定网络环境）")
    except Exception as e:
        print(f"[行业板块实时] 获取失败: {e}")

    # 测试概念板块实时行情
    print("\n" + "=" * 50)
    print("测试概念板块实时行情获取")
    print("=" * 50)
    try:
        quote = fetcher.get_realtime_quote('003490', stock_type='concept', stock_name='军工')
        if quote:
            print(f"[概念板块实时] {quote.name}: 价格={quote.price}, 涨跌幅={quote.change_pct}%")
        else:
            print("[概念板块实时] 未获取到数据（可能需要特定网络环境）")
    except Exception as e:
        print(f"[概念板块实时] 获取失败: {e}")
