"""Export SUMO road network to GeoJSON for frontend replay and map rendering."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import sumolib


def export_sumo_network_to_geojson(
    net_file: str | Path,
    output_file: str | Path = "data/processed/network.geojson",
    use_geo_coords: bool = True
) -> Dict[str, Any]:
    """
    Parse a SUMO network file and convert roads, junctions, and TLS to GeoJSON.
    
    Args:
        net_file: Path to ingolstadt.net.xml.
        output_file: Output path for network.geojson.
        use_geo_coords: If True, uses longitude/latitude if projection is present in net.xml.
                       If False, uses raw SUMO cartesian coordinates (x, y).
                       
    Returns:
        GeoJSON FeatureCollection dict.
    """
    net_path = Path(net_file)
    if not net_path.exists():
        raise FileNotFoundError(f"Network file not found: {net_path}")

    net = sumolib.net.readNet(str(net_path))
    features: List[Dict[str, Any]] = []

    # Test projection availability
    has_proj = False
    if use_geo_coords:
        try:
            net.getGeoProj()
            has_proj = True
        except Exception:
            has_proj = False
    for edge in net.getEdges():
        if edge.isSpecial():
            continue  # skip internal edges if desired

        for lane in edge.getLanes():
            shape = lane.getShape()
            coords = []
            for x, y in shape:
                if has_proj:
                    lon, lat = net.convertXY2LonLat(x, y)
                    coords.append([lon, lat])
                else:
                    coords.append([x, y])

            feature = {
                "type": "Feature",
                "id": lane.getID(),
                "properties": {
                    "type": "lane",
                    "edge_id": edge.getID(),
                    "speed": lane.getSpeed(),
                    "length": lane.getLength(),
                    "width": lane.getWidth(),
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": coords
                }
            }
            features.append(feature)

    # 2. Export Junctions & Traffic Lights
    for node in net.getNodes():
        x, y = node.getCoord()
        if has_proj:
            coord = list(net.convertXY2LonLat(x, y))
        else:
            coord = [x, y]

        is_tls = node.getType() == "traffic_light"
        tls_id = node.getTLSID() if is_tls else None

        feature = {
            "type": "Feature",
            "id": node.getID(),
            "properties": {
                "type": "junction",
                "node_type": node.getType(),
                "is_tls": is_tls,
                "tls_id": tls_id
            },
            "geometry": {
                "type": "Point",
                "coordinates": coord
            }
        }
        features.append(feature)

    geojson = {
        "type": "FeatureCollection",
        "features": features
    }

    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(geojson, f)

    return geojson


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Export SUMO network to GeoJSON")
    parser.add_argument("--net", default="data/raw/intas/scenario/ingolstadt.net.xml")
    parser.add_argument("--out", default="data/processed/network.geojson")
    parser.add_argument("--cartesian", action="store_true", help="Use cartesian x,y instead of lon/lat")
    args = parser.parse_args()

    export_sumo_network_to_geojson(args.net, args.out, use_geo_coords=not args.cartesian)
    print(f"Network exported to {args.out}")
