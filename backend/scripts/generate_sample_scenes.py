import os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

os.makedirs("data/sample_scenes", exist_ok=True)

def generate_lake_pichola():
    """Lake Pichola: Deep blue/cyan water bodies nestled in ochre urban and green Aravalli hills"""
    w, h = 600, 600
    img = Image.new("RGB", (w, h), color=(185, 172, 142)) # Ochre/sand urban background
    draw = ImageDraw.Draw(img)

    # Surrounding vegetation and hills (green/olive patches)
    for _ in range(80):
        cx, cy = np.random.randint(0, w), np.random.randint(0, h)
        rad = np.random.randint(30, 90)
        draw.ellipse([cx-rad, cy-rad, cx+rad, cy+rad], fill=(78, 112, 62))

    # Lake Pichola main water body (deep cyan/blue)
    lake_poly = [
        (180, 120), (260, 150), (320, 220), (350, 340), (330, 450),
        (280, 520), (220, 480), (190, 410), (160, 330), (150, 220)
    ]
    draw.polygon(lake_poly, fill=(24, 75, 120))

    # Fateh Sagar secondary water body
    fateh_poly = [(360, 80), (440, 100), (460, 180), (410, 240), (350, 190)]
    draw.polygon(fateh_poly, fill=(28, 85, 135))

    # Dense urban road grid & ghats
    for y in range(80, 550, 35):
        draw.line([(50, y), (550, y)], fill=(210, 205, 195), width=2)
    for x in range(80, 550, 35):
        draw.line([(x, 50), (x, 550)], fill=(210, 205, 195), width=2)

    # Blur slightly for natural satellite optical texture
    img = img.filter(ImageFilter.GaussianBlur(1.2))
    img.save("data/sample_scenes/lake_pichola_s2.png")

def generate_hussain_sagar():
    """Hussain Sagar: Heart-shaped deep navy lake surrounded by dense metropolitan grid and greenery"""
    w, h = 600, 600
    img = Image.new("RGB", (w, h), color=(165, 160, 155)) # Dense concrete gray
    draw = ImageDraw.Draw(img)

    # Urban grid
    for y in range(40, 580, 25):
        draw.line([(20, y), (580, y)], fill=(195, 190, 180), width=2)
    for x in range(40, 580, 25):
        draw.line([(x, 20), (x, 580)], fill=(195, 190, 180), width=2)

    # Urban parks (Necklace Road, Sanjeevaiah Park)
    draw.rectangle([120, 140, 220, 460], fill=(65, 115, 55))
    draw.rectangle([380, 160, 480, 300], fill=(70, 120, 60))

    # Hussain Sagar Lake (distinct heart-like geometry)
    lake_points = [
        (280, 180), (340, 170), (400, 210), (410, 290), (370, 370),
        (300, 430), (240, 380), (220, 290), (240, 210)
    ]
    draw.polygon(lake_points, fill=(18, 60, 105))

    img = img.filter(ImageFilter.GaussianBlur(1.0))
    img.save("data/sample_scenes/hussain_sagar_s2.png")

def generate_sabarmati():
    """Sabarmati River: Diagonal water ribbon with embankments, bridges, and city grid"""
    w, h = 600, 600
    img = Image.new("RGB", (w, h), color=(175, 168, 155))
    draw = ImageDraw.Draw(img)

    # City blocks
    for y in range(30, 580, 30):
        draw.line([(20, y), (580, y)], fill=(200, 195, 185), width=2)
    for x in range(30, 580, 30):
        draw.line([(x, 20), (x, 580)], fill=(200, 195, 185), width=2)

    # Sabarmati River corridor (diagonal swath)
    river_coords = [
        (120, 0), (220, 0), (460, 600), (360, 600)
    ]
    draw.polygon(river_coords, fill=(30, 90, 130))

    # Riverfront promenade / green verge
    draw.line([(120, 0), (360, 600)], fill=(70, 130, 60), width=8)
    draw.line([(220, 0), (460, 600)], fill=(70, 130, 60), width=8)

    # Major bridges
    for pos in [150, 300, 450]:
        draw.line([(pos-50, pos-20), (pos+60, pos+30)], fill=(235, 235, 235), width=5)

    img = img.filter(ImageFilter.GaussianBlur(1.0))
    img.save("data/sample_scenes/sabarmati_ahmedabad_s2.png")

def generate_sundarbans():
    """Sundarbans: Deep emerald mangrove wetland interwoven with muddy estuarine channels"""
    w, h = 600, 600
    img = Image.new("RGB", (w, h), color=(38, 85, 45)) # Dense mangrove green
    draw = ImageDraw.Draw(img)

    # Intertwined tidal channels (turbid brackish blue/brown water)
    for i in range(12):
        points = []
        cx = np.random.randint(50, 550)
        for y in range(0, 600, 40):
            cx += np.random.randint(-25, 25)
            points.append((cx, y))
        draw.line(points, fill=(45, 95, 120), width=np.random.randint(15, 45))

    img = img.filter(ImageFilter.GaussianBlur(1.5))
    img.save("data/sample_scenes/sundarbans_delta_s2.png")

def generate_bhadla():
    """Bhadla Solar Park: Ochre desert sands with vast blue/dark rectangular photovoltaic arrays"""
    w, h = 600, 600
    img = Image.new("RGB", (w, h), color=(220, 195, 145)) # Desert sand
    draw = ImageDraw.Draw(img)

    # Desert dunes / terrain gradients
    for y in range(0, 600, 20):
        draw.line([(0, y), (600, y)], fill=(210, 185, 135), width=3)

    # Solar PV panel array blocks (deep reflective blue-black grid)
    for row in range(4):
        for col in range(5):
            x1 = 70 + col * 95
            y1 = 80 + row * 110
            x2 = x1 + 75
            y2 = y1 + 80
            draw.rectangle([x1, y1, x2, y2], fill=(22, 35, 60), outline=(190, 190, 190), width=1)
            # Internal sub-panel lines
            for sub_y in range(y1 + 10, y2, 10):
                draw.line([(x1, sub_y), (x2, sub_y)], fill=(35, 55, 90), width=1)

    img = img.filter(ImageFilter.GaussianBlur(0.8))
    img.save("data/sample_scenes/bhadla_solar_s2.png")

if __name__ == "__main__":
    generate_lake_pichola()
    generate_hussain_sagar()
    generate_sabarmati()
    generate_sundarbans()
    generate_bhadla()
    print("Sample scenes successfully generated.")
