#   ____  _ _        _      ____  _  _     ____  _____ ____ _____
#  | __ )(_) |_ _ __(_)_  _|___ \| || |   |  _ \| ____/ ___|_   _|
#  |  _ \| | __| '__| \ \/ / __) | || |_  | |_) |  _| \___ \ | |
#  | |_) | | |_| |  | |>  < / __/|__   _| |  _ <| |___ ___) || |
#  |____/|_|\__|_|  |_/_/\_\_____|  |_|   |_| \_\_____|____/ |_|

import asyncio
import itertools
import ssl
import warnings
from typing import Any, Dict, Optional
from urllib.parse import urlparse

from aiohttp import ClientSession, TCPConnector
from loguru import logger

from .exceptions import BitrixError


class Bitrix24:
    """
    Bitrix24 API class.

    Provides an easy way to communicate with Bitrix24 portal over REST without OAuth.
    """

    def __init__(
        self,
        domain: str,
        timeout: int = 60,
        safe: bool = True,
        fetch_all_pages: bool = True,
        retry_after: int = 3,
    ):
        """
        Create Bitrix24 API object.

        Parameters
        ----------
            domain (str): Bitrix24 webhook domain
            timeout (int): Timeout for API request in seconds
            safe (bool): Set to `False` to ignore the certificate verification
            fetch_all_pages (bool): Fetch all pages for paginated requests
            retry_after (int): Retry after seconds for QUERY_LIMIT_EXCEEDED error
        """
        self._domain = self._prepare_domain(domain)
        self._timeout = int(timeout)
        self._fetch_all_pages = bool(fetch_all_pages)
        self._retry_after = int(retry_after)
        self._verify_ssl = bool(safe)
        logger.info(f"Bitrix24 API initialized with domain: {self._domain}")

    @staticmethod
    def _prepare_domain(domain: str) -> str:
        """Normalize user passed domain to a valid one."""
        o = urlparse(domain)
        if not o.scheme or not o.netloc:
            logger.error(f"Invalid domain provided: {domain}")
            raise BitrixError("Not a valid domain. Please provide a valid domain.")
        user_id, code = o.path.split("/")[2:4]
        prepared_domain = "{0}://{1}/rest/{2}/{3}".format(
            o.scheme, o.netloc, user_id, code
        )
        logger.debug(f"Prepared domain: {prepared_domain}")
        return prepared_domain

    def _prepare_params(self, params: Dict[str, Any], prev: str = "") -> str:
        """
        Transform list of parameters to a valid bitrix array.

        Parameters
        ----------
            params (dict): Dictionary of parameters
            prev (str): Previous key

        Returns
        -------
            str: Prepared parameters
        """
        ret = ""
        if isinstance(params, dict):
            for key, value in params.items():
                if isinstance(value, dict):
                    if prev:
                        key = "{0}[{1}]".format(prev, key)
                    ret += self._prepare_params(value, key)
                elif (isinstance(value, list) or isinstance(value, tuple)) and len(
                    value
                ) > 0:
                    for offset, val in enumerate(value):
                        if isinstance(val, dict):
                            ret += self._prepare_params(
                                val, "{0}[{1}][{2}]".format(prev, key, offset)
                            )
                        else:
                            if prev:
                                ret += "{0}[{1}][{2}]={3}&".format(
                                    prev, key, offset, val
                                )
                            else:
                                ret += "{0}[{1}]={2}&".format(key, offset, val)
                else:
                    if prev:
                        ret += "{0}[{1}]={2}&".format(prev, key, value)
                    else:
                        ret += "{0}={1}&".format(key, value)
        logger.debug(f"Prepared parameters: {ret}")
        return ret

    async def request(
        self, method: str, params: Optional[str] = None
    ) -> Dict[str, Any]:
        ssl_context = ssl.create_default_context()
        if not self._verify_ssl:
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            logger.warning("SSL verification is disabled")
        async with ClientSession(connector=TCPConnector(ssl=ssl_context)) as session:
            url = f"{self._domain}/{method}.json"
            logger.info(f"Making request to: {url}")
            async with session.get(url, params=params, timeout=self._timeout) as resp:
                logger.info(f"Response received with status: {resp.status}")
                logger.debug(f"Response text: {await resp.text()}")
                if resp.status not in [200, 201]:
                    logger.error(f"HTTP error: {resp.status}")
                    raise BitrixError(f"HTTP error: {resp.status}")
                response = await resp.json()
                if "error" in response:
                    if response["error"] == "QUERY_LIMIT_EXCEEDED":
                        logger.warning(
                            f"Query limit exceeded. Retrying after {self._retry_after} seconds"
                        )
                        await asyncio.sleep(self._retry_after)
                        return await self.request(method, params)
                    logger.error(f"Bitrix error: {response['error_description']}")
                    raise BitrixError(response["error_description"], response["error"])
                logger.debug(f"Response received: {response}")
                return response

    async def _call(
        self, method: str, params: Dict[str, Any] = None, start: int = 0
    ) -> Dict[str, Any]:
        """Async call a REST method with specified parameters.

        Parameters
        ----------
            method (str): REST method name
            params (dict): Optional arguments which will be converted to a POST request string
            start (int): Offset for pagination
        """
        if params is None:
            params = {}
        params["start"] = start

        payload = self._prepare_params(params)
        logger.info(f"Calling method: {method} with start: {start}")
        res = await self.request(method, payload)

        if "next" in res and not start and self._fetch_all_pages:
            if res["total"] % 50 == 0:
                count_tasks = res["total"] // 50 - 1
            else:
                count_tasks = res["total"] // 50

            logger.info(
                f"Fetching all pages for method: {method}. Total tasks: {count_tasks}"
            )
            tasks = [
                self._call(method, params, (s + 1) * 50) for s in range(count_tasks)
            ]
            items = await asyncio.gather(*tasks)
            if type(res["result"]) is not dict:
                logger.debug("Combining results from all pages")
                return res["result"] + list(itertools.chain(*items))
            if items:
                key = list(res["result"].keys())[0]
                for item in items:
                    res["result"][key] += item[key]
        return res["result"]

    def callMethod(
        self, method: str, params: Dict[str, Any] = None, **kwargs
    ) -> Dict[str, Any]:
        """Call a REST method with specified parameters.

        Parameters
        ----------
            method (str): REST method name
            params (dict): Optional arguments which will be converted to a POST request string

        Returns
        -------
            Returning the REST method response as an array, an object or a scalar
        """
        if params is None:
            params = {}

        if not method:
            logger.error("Wrong method name provided")
            raise BitrixError("Wrong method name", 400)

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            warnings.warn(
                "You are using `callMethod` method in a synchronous way. "
                "Starting from version 3, this method will be completely asynchronous."
                "Please consider updating your code",
                DeprecationWarning,
            )
            logger.warning(
                "callMethod is being used synchronously. This will change in version 3."
            )
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(self._call(method, params or kwargs))
            finally:
                loop.close()
        else:
            logger.info(f"Calling method asynchronously: {method}")
            result = asyncio.ensure_future(self._call(method, params or kwargs))
        return result
