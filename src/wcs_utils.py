import time
from dataclasses import dataclass, field
from xml.etree import ElementTree as ET

import requests

WCS_BASE = "https://environment.data.gov.uk/spatialdata/lidar-composite-digital-terrain-model-dtm-1m/wcs"
EPSG_27700 = "http://www.opengis.net/def/crs/EPSG/0/27700"

WCS_NS = "{http://www.opengis.net/wcs/2.0}"
GML_NS = "{http://www.opengis.net/gml/3.2}"


class WCSError(Exception): ...


@dataclass
class Coverage:
    """server specific request fields, to print at runtime"""

    coverage_id: str
    axis_x: str
    axis_y: str
    tiff_format: str


@dataclass
class WCSClient:
    base: str = WCS_BASE
    version: str = "2.0.1"
    timeout: int = 120
    max_retries: int = 5
    backoff: float = 2.0
    session: requests.Session = field(default_factory=requests.Session)

    def _get(self, params: list[tuple[str, str]]) -> bytes:

        full = [("service", "WCS"), ("version", self.version), *params]
        last: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                r = self.session.get(self.base, params=full, timeout=self.timeout)
                if r.status_code == 200:
                    return r.content

                # use wcs error for retryable codes, let 400 codes crash
                if r.status_code < 500 and r.status_code != 429:
                    raise WCSError(f"{r.status_code} on {r.url}\n{r.text[:500]}")
                last = WCSError(f"{r.status_code} on {r.url}")

            except requests.RequestException as e:
                last = e
            time.sleep(self.backoff * (2**attempt))

        raise WCSError(f"exhausted {self.max_retries} retries: {last}")

    def discover(self) -> Coverage:

        caps = ET.fromstring(self._get([("request", "GetCapabilities")]))
        ids = [e.text for e in caps.iter(f"{WCS_NS}CoverageId") if e.text]

        if not ids:
            raise WCSError("no CoverageId advertised")
        coverage_id = next(
            (i for i in ids if "elevation" in i.lower() or "dtm" in i.lower()), ids[0]
        )

        formats = [e.text for e in caps.iter(f"{WCS_NS}formatSupported") if e.text]
        tiff = next((f for f in formats if "tiff" in f.lower()), None)

        if tiff is None:
            raise WCSError(f"no geotiff format advertised: {formats}")

        desc = ET.fromstring(
            self._get([("request", "DescribeCoverage"), ("coverageId", coverage_id)])
        )

        labels = next(
            (
                env.attrib["axisLabels"]
                for env in desc.iter(f"{GML_NS}Envelope")
                if "axisLabels" in env.attrib
            ),
            None,
        )

        if not labels:
            raise WCSError("no axisLabels in DescribeCoverage envelope")

        axis_x, axis_y = labels.split()[:2]

        return Coverage(coverage_id, axis_x, axis_y, tiff)

    def get_coverage(
        self, cov: Coverage, bbox: tuple[float, float, float, float]
    ) -> bytes:
        """bbox = (xmin, ymin, xmax, ymax) in epsg:27700 -> geotiff bytes."""

        xmin, ymin, xmax, ymax = bbox
        return self._get(
            [
                ("request", "GetCoverage"),
                ("coverageId", cov.coverage_id),
                ("format", cov.tiff_format),
                ("subset", f"{cov.axis_x}({xmin},{xmax})"),
                ("subset", f"{cov.axis_y}({ymin},{ymax})"),
                ("SUBSETTINGCRS", EPSG_27700),
                ("OUTPUTCRS", EPSG_27700),
            ]
        )
