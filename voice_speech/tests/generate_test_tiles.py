"""Generate realistic synthetic Sentinel-2 style RGB test tiles for VQA evaluation."""

import os
from pathlib import Path
from PIL import Image, ImageDraw


def create_sample_tiles(output_dir: str = "voice_speech/data/samples") -> dict[str, str]:
    """Generates 4 distinct satellite test tiles matching the 5 question categories:
    1. Coastal / Water tile (Sentinel-2 blue water + coast)
    2. Urban / City tile (built-up structures, roads)
    3. Agricultural / Forest tile (vegetation, fields)
    4. Port / Maritime tile (harbor with visible vessels)
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    tiles = {}

    # Tile 1: Coastal Water
    img_water = Image.new("RGB", (256, 256), color=(20, 60, 140))
    draw = ImageDraw.Draw(img_water)
    # Add coastline
    draw.polygon([(0, 180), (120, 160), (256, 200), (256, 256), (0, 256)], fill=(120, 140, 80))
    p1 = out_path / "sentinel2_coastal_water.png"
    img_water.save(p1)
    tiles["water"] = str(p1)

    # Tile 2: Urban Built-up
    img_urban = Image.new("RGB", (256, 256), color=(140, 135, 130))
    draw = ImageDraw.Draw(img_urban)
    # Draw road grid
    draw.line([(0, 128), (256, 128)], fill=(60, 60, 60), width=6)
    draw.line([(128, 0), (128, 256)], fill=(60, 60, 60), width=6)
    # Draw building blocks
    for x in range(20, 240, 40):
        for y in range(20, 240, 40):
            if abs(x - 128) > 20 and abs(y - 128) > 20:
                draw.rectangle([x, y, x + 25, y + 25], fill=(180, 70, 60), outline=(220, 220, 220))
    p2 = out_path / "sentinel2_urban_grid.png"
    img_urban.save(p2)
    tiles["urban"] = str(p2)

    # Tile 3: Vegetation Forest
    img_veg = Image.new("RGB", (256, 256), color=(35, 120, 45))
    draw = ImageDraw.Draw(img_veg)
    # Draw agricultural parcel boundaries
    draw.line([(0, 80), (256, 80)], fill=(70, 140, 50), width=3)
    draw.line([(100, 0), (100, 256)], fill=(70, 140, 50), width=3)
    p3 = out_path / "sentinel2_forest_agriculture.png"
    img_veg.save(p3)
    tiles["vegetation"] = str(p3)

    # Tile 4: Port & Maritime Ships
    img_port = Image.new("RGB", (256, 256), color=(25, 75, 160))
    draw = ImageDraw.Draw(img_port)
    # Pier / dock
    draw.rectangle([0, 100, 160, 140], fill=(130, 130, 130))
    # Ships / vessels
    draw.polygon([(180, 90), (210, 85), (210, 95)], fill=(240, 240, 240))
    draw.polygon([(190, 150), (220, 145), (220, 155)], fill=(240, 240, 240))
    p4 = out_path / "sentinel2_maritime_port.png"
    img_port.save(p4)
    tiles["port"] = str(p4)

    return tiles


if __name__ == "__main__":
    generated = create_sample_tiles()
    for name, path in generated.items():
        print(f"Generated sample tile [{name}]: {path}")
