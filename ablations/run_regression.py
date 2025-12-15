#!/usr/bin/env python3
"""End-to-end heat demand estimation pipeline for Kaiserslautern."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import osmnx as ox
import pandas as pd
from shapely import wkt
from shapely.geometry import LineString, MultiLineString
from shapely.ops import polygonize, unary_union
from shapely.validation import make_valid

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOGGER = logging.getLogger(__name__)

DEFAULT_BBOX = "7.711388,49.410640,7.812840,49.468572"
WFS_BASE_URL = "https://www.energieatlas.rlp.de/geoserver/earp/ows"
WFS_LAYER = "wb_rlp_agg_gmndesch10_4326"
LOD2_PATH = Path("lod2_buildings_height.gpkg")
LOD2_LAYER = "building"
CENSUS_CSV = Path("Kaiserslautern.csv")
DEFAULT_FLOOR_HEIGHT = 3.0
SYSTEM_EFFICIENCY = 0.9
ENERGY_COLUMNS = [
    "id_agg",
    "wb_gs",
    "geb_n",
    "Shape_Leng",
    "flaeche",
    "Shape_Area",
]

TABULA_Q_H_DE = {
    "SFH": {"A": 220, "B": 210, "C": 200, "D": 190, "E": 180, "F": 140, "G": 110, "H": 90},
    "TH": {"A": 200, "B": 190, "C": 180, "D": 170, "E": 150, "F": 130, "G": 105, "H": 85},
    "MFH": {"A": 190, "B": 180, "C": 170, "D": 160, "E": 145, "F": 125, "G": 100, "H": 80},
    "NR": {"A": 150, "B": 140, "C": 130, "D": 120, "E": 110, "F": 95, "G": 85, "H": 75},
}


@dataclass
class PipelineConfig:
    """Configuration bundle for the heat demand pipeline."""

    bbox: str = DEFAULT_BBOX
    wfs_base_url: str = WFS_BASE_URL
    wfs_layer: str = WFS_LAYER
    lod2_path: Path = LOD2_PATH
    lod2_layer: str = LOD2_LAYER
    census_csv: Path = CENSUS_CSV
    floor_height_m: float = DEFAULT_FLOOR_HEIGHT


def build_wfs_url(config: PipelineConfig) -> str:
    """Return a WFS GetFeature URL for the configured bounding box."""

    return (
        f"{config.wfs_base_url}?service=WFS&version=1.1.0&request=GetFeature"
        f"&typeName={config.wfs_layer}&outputFormat=application/json"
        "&srsName=EPSG:4326"
        f"&bbox={config.bbox},EPSG:4326"
    )


def fetch_roi(config: PipelineConfig) -> gpd.GeoDataFrame:
    """Fetch the region of interest polygons from Energieatlas."""

    wfs_url = build_wfs_url(config)
    LOGGER.info("Fetching ROI polygons from %s", wfs_url)
    roi = gpd.read_file(wfs_url).to_crs(epsg=4326)
    roi["geometry"] = roi.geometry.apply(make_valid)
    return roi


def union_roi_geometry(roi: gpd.GeoDataFrame):
    """Return a union geometry for querying OSM."""

    if roi.empty:
        raise ValueError("ROI GeoDataFrame is empty")
    return roi.geometry.union_all()


def fetch_osm_buildings(geometry) -> gpd.GeoDataFrame:
    """Pull OSM buildings intersecting the supplied geometry."""

    LOGGER.info("Fetching OSM buildings for ROI union polygon")
    buildings = ox.features_from_polygon(geometry, tags={"building": True})
    buildings = buildings[buildings.geometry.type.isin(["Polygon", "MultiPolygon"])].reset_index()
    buildings = buildings.rename(columns={"osmid": "id"})
    return buildings


def attach_energy_attributes(buildings: gpd.GeoDataFrame, roi: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Spatially join Energieatlas attributes to each building polygon."""

    cols = ["geometry"] + [col for col in ENERGY_COLUMNS if col in roi.columns]
    roi_energy = roi[cols].copy()
    joined = gpd.sjoin(buildings, roi_energy, how="left", predicate="within")
    if "index_right" in joined.columns:
        joined = joined.drop(columns="index_right")
    if "id_agg" in joined.columns:
        joined = joined[joined["id_agg"].notna()].reset_index(drop=True)
    return joined


def multilinez_to_polygon(geom):
    """Convert a (multi)linestring 3D footprint to a 2D polygon."""

    if geom is None or (isinstance(geom, float) and np.isnan(geom)):
        return None
    if isinstance(geom, str):
        geom = wkt.loads(geom)
    if geom.is_empty:
        return None

    line_strs: list[LineString] = []
    if isinstance(geom, LineString):
        coords = [(x, y) for x, y, *_ in geom.coords]
        line_strs.append(LineString(coords))
    elif isinstance(geom, MultiLineString):
        for part in geom.geoms:
            coords = [(x, y) for x, y, *_ in part.coords]
            line_strs.append(LineString(coords))
    else:
        return None

    merged = unary_union(line_strs)
    polygons = list(polygonize(merged))
    if not polygons:
        return None
    return max(polygons, key=lambda poly: poly.area)


def load_lod2_footprints(config: PipelineConfig, crs) -> gpd.GeoDataFrame:
    """Load LOD2 buildings, convert roof wireframes to footprint polygons."""

    if not config.lod2_path.exists():
        LOGGER.warning("LOD2 dataset %s not found", config.lod2_path)
        return gpd.GeoDataFrame(columns=["geometry"], crs=crs)

    LOGGER.info("Loading LOD2 dataset from %s", config.lod2_path)
    lod2 = gpd.read_file(config.lod2_path, layer=config.lod2_layer)
    lod2 = lod2.to_crs(crs)
    lod2["footprint"] = lod2.geometry.apply(multilinez_to_polygon)
    lod2 = lod2.dropna(subset=["footprint"])
    lod2 = lod2.set_geometry("footprint", drop=True).rename_geometry("geometry")
    return lod2


def link_lod2_heights(buildings: gpd.GeoDataFrame, lod2: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Assign mean measured height per building from the LOD2 dataset."""

    if lod2.empty:
        buildings = buildings.copy()
        buildings["measuredHeight"] = np.nan
        return buildings

    lod2_cent = lod2.copy()
    lod2_cent["centroid"] = lod2_cent.geometry.centroid
    lod2_cent = lod2_cent.set_geometry("centroid")

    joined = gpd.sjoin(lod2_cent, buildings[["id", "geometry"]], how="left", predicate="within")
    if "index_right" in joined.columns:
        joined = joined.drop(columns="index_right")

    height = (
        joined.dropna(subset=["id"])
        .groupby("id")["measuredHeight"]
        .mean()
        .rename("measuredHeight")
    )

    buildings = buildings.copy()
    buildings["measuredHeight"] = buildings["id"].map(height)
    return buildings


def fill_missing_heights(buildings: gpd.GeoDataFrame, epsg: int = 25832) -> gpd.GeoDataFrame:
    """Fill unknown heights using nearest neighbour imputation."""

    buildings = buildings.copy()
    if "measuredHeight" not in buildings.columns:
        buildings["measuredHeight"] = np.nan

    known = buildings[buildings["measuredHeight"].notna()]
    unknown = buildings[buildings["measuredHeight"].isna()]

    if known.empty or unknown.empty:
        return buildings

    gdf_proj = buildings.to_crs(epsg=epsg)
    known_proj = gdf_proj.loc[known.index]
    unknown_proj = gdf_proj.loc[unknown.index]

    matched = gpd.sjoin_nearest(
        unknown_proj[["geometry"]],
        known_proj[["geometry", "measuredHeight"]],
        how="left",
        distance_col="dist_to_nearest",
    )

    buildings.loc[matched.index, "measuredHeight"] = matched["measuredHeight"].values
    return buildings


def load_census_age(config: PipelineConfig) -> gpd.GeoDataFrame:
    """Load census building age data as a GeoDataFrame."""

    if not config.census_csv.exists():
        LOGGER.warning("Census CSV %s not found", config.census_csv)
        return gpd.GeoDataFrame(columns=["geometry"], crs="EPSG:3035")

    LOGGER.info("Loading census age data from %s", config.census_csv)
    df = pd.read_csv(config.census_csv, sep=",", encoding="latin1")
    age = df[df["Merkmal"] == "BAUJAHR_MZ"].copy()
    age = age.dropna(subset=["x_3035", "y_3035"])
    age = gpd.GeoDataFrame(
        age,
        geometry=gpd.points_from_xy(age["x_3035"], age["y_3035"]),
        crs="EPSG:3035",
    )
    return age


def attach_building_age(buildings: gpd.GeoDataFrame, census: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Map each building to the nearest census age point."""

    if census.empty:
        buildings = buildings.copy()
        buildings["building_age_group"] = None
        return buildings

    buildings_3035 = buildings.to_crs(census.crs)
    matched = gpd.sjoin_nearest(
        buildings_3035,
        census[["geometry", "Auspraegung_Text", "Gitter_ID_100m"]],
        how="left",
        distance_col="dist_to_age_point",
    )

    matched = matched.rename(columns={"Auspraegung_Text": "building_age_group"})
    matched = matched.to_crs(buildings.crs)
    selection = buildings.copy()
    new_columns = [col for col in matched.columns if col not in selection.columns]
    for col in new_columns:
        selection[col] = matched[col]
    selection["building_age_group"] = matched["building_age_group"].values
    selection["dist_to_age_point"] = matched.get("dist_to_age_point")
    selection["Gitter_ID_100m"] = matched.get("Gitter_ID_100m")
    return selection


def ageband_to_tabula_era(age_group: str | None) -> str | None:
    """Map census age bands to TABULA-DE era letters."""

    if not isinstance(age_group, str):
        return None
    band = age_group.strip()
    if band == "Vor 1919":
        return "A"
    if band == "1919 - 1948":
        return "B"
    if band == "1949 - 1978":
        return "E"
    if band == "1979 - 1986":
        return "F"
    if band == "1987 - 1990":
        return "G"
    if band == "1991 - 1995":
        return "G"
    if band in {"1996 - 2000", "2001 - 2004", "2005 - 2008", "2009 und später"}:
        return "H"
    return None


def map_osm_to_tabula_type(osm_building_value: str | None) -> str | None:
    """Map raw OSM building tags to coarse TABULA categories."""

    tag = (osm_building_value or "").lower()
    if tag in {"house", "detached"}:
        return "SFH"
    if tag in {"semi", "semidetached_house", "terrace", "row_house"}:
        return "TH"
    if tag in {"apartments", "block", "residential", "dormitory"}:
        return "MFH"
    if tag in {
        "office",
        "commercial",
        "retail",
        "industrial",
        "warehouse",
        "school",
        "university",
        "hospital",
        "public",
        "church",
    }:
        return "NR"
    return None


def classify_tabula_type_with_fallback(osm_building_value: str | None, footprint_area_m2: float) -> str | None:
    """Use OSM tags or footprint size to decide on a TABULA building type."""

    candidate = map_osm_to_tabula_type(osm_building_value)
    if candidate is not None:
        return candidate
    if pd.notna(footprint_area_m2) and footprint_area_m2 < 250:
        return "SFH"
    if pd.notna(footprint_area_m2):
        return "MFH"
    return None


def add_floor_area_from_height(gdf: gpd.GeoDataFrame, floor_height: float = DEFAULT_FLOOR_HEIGHT) -> gpd.GeoDataFrame:
    """Compute footprint, floors, and total floor area from geometry and heights."""

    if "geometry" not in gdf.columns:
        raise ValueError("GeoDataFrame requires a geometry column")

    result = gdf.copy()
    result["measuredHeight"] = pd.to_numeric(result.get("measuredHeight"), errors="coerce")
    projected = result.to_crs(25832)
    result["footprint_area_m2"] = projected.geometry.area

    floors = (result["measuredHeight"] / floor_height).round()
    floors = floors.clip(lower=1).fillna(1).astype(int)
    result["n_floors"] = floors
    result["floor_area_m2"] = result["footprint_area_m2"] * result["n_floors"]
    return result


def lookup_tabula_q_h(building_type: str | None, era: str | None) -> float:
    """Return specific heat demand per TABULA category."""

    if building_type is None or era is None:
        return np.nan
    return TABULA_Q_H_DE.get(building_type, {}).get(era, np.nan)


def apply_tabula_de(buildings: gpd.GeoDataFrame, floor_height: float = DEFAULT_FLOOR_HEIGHT) -> gpd.GeoDataFrame:
    """Compute TABULA-DE style heat demand metrics for each building."""

    enriched = buildings.copy()
    if "building" not in enriched.columns:
        enriched["building"] = None
    if "building_age_group" not in enriched.columns:
        enriched["building_age_group"] = None

    enriched = add_floor_area_from_height(enriched, floor_height=floor_height)
    enriched["tabula_era"] = enriched["building_age_group"].apply(ageband_to_tabula_era)
    enriched["tabula_type"] = [
        classify_tabula_type_with_fallback(tag, area)
        for tag, area in zip(enriched["building"], enriched["footprint_area_m2"])
    ]
    enriched["q_h_kwh_m2a"] = [
        lookup_tabula_q_h(btype, era)
        for btype, era in zip(enriched["tabula_type"], enriched["tabula_era"])
    ]
    enriched["Q_h_kwh_a"] = enriched["q_h_kwh_m2a"] * enriched["floor_area_m2"]
    enriched["Q_final_kwh_a"] = enriched["Q_h_kwh_a"] / SYSTEM_EFFICIENCY
    return enriched


def aggregate_energy(buildings: gpd.GeoDataFrame) -> pd.DataFrame | None:
    """Aggregate heat metrics by Energieatlas cell if identifiers exist."""

    if "id_agg" not in buildings.columns:
        return None
    summary = (
        buildings.groupby("id_agg")
        .agg(
            n_buildings=("id", "count"),
            total_floor_area_m2=("floor_area_m2", "sum"),
            total_Q_h_kwh_a=("Q_h_kwh_a", "sum"),
            total_Q_final_kwh_a=("Q_final_kwh_a", "sum"),
            mean_q_h_kwh_m2a=("q_h_kwh_m2a", "mean"),
        )
        .reset_index()
    )
    return summary


def run_pipeline(config: PipelineConfig) -> tuple[gpd.GeoDataFrame, pd.DataFrame | None]:
    """Execute the entire heat demand estimation pipeline."""

    roi = fetch_roi(config)
    roi_union = union_roi_geometry(roi)
    buildings = fetch_osm_buildings(roi_union)
    buildings = attach_energy_attributes(buildings, roi)

    lod2 = load_lod2_footprints(config, buildings.crs)
    buildings = link_lod2_heights(buildings, lod2)
    buildings = fill_missing_heights(buildings)

    census = load_census_age(config)
    buildings = attach_building_age(buildings, census)

    buildings = apply_tabula_de(buildings, floor_height=config.floor_height_m)
    summary = aggregate_energy(buildings)
    return buildings, summary


def preview_outputs(buildings: gpd.GeoDataFrame, summary: pd.DataFrame | None, limit: int = 5) -> None:
    """Log quick previews of the detailed and aggregated outputs."""

    LOGGER.info("First %d buildings:\n%s", limit, buildings.head(limit))
    if summary is not None:
        LOGGER.info("First %d Energieatlas cells:\n%s", limit, summary.head(limit))
    else:
        LOGGER.info("No Energieatlas aggregation available; id_agg column missing")


def main() -> None:
    """Run the pipeline with default configuration and preview results."""

    config = PipelineConfig()
    buildings, summary = run_pipeline(config)
    preview_outputs(buildings, summary)


if __name__ == "__main__":
    main()