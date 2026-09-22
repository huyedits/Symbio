"""Blob segmentation: what stands out on a screen, before anything names it.

No model is loaded here and no screen is grabbed. Every image is drawn in the
test, so what is a control and what is wallpaper is known rather than assumed
— which is the whole claim this module makes.
"""

import pytest

from symbio import blobs

pytest.importorskip("PIL", reason="Pillow is what does the arithmetic")


def _screen(tmp_path, name="screen.png", size=(1200, 800)):
    """A window-less desktop: a gradient wallpaper and nothing else."""
    from PIL import Image, ImageDraw

    im = Image.new("RGB", size)
    draw = ImageDraw.Draw(im)
    for y in range(size[1]):
        draw.line([(0, y), (size[0], y)], fill=(40 + y // 12, 60 + y // 20, 90))
    path = tmp_path / name
    im.save(path)
    return im, draw, path


def _save(im, path):
    im.save(path)
    return str(path)


def _contains(regions, box, slack=20):
    """Is one of the regions sitting on this rectangle?"""
    x1, y1, x2, y2 = box
    for region in regions:
        rx1, ry1, rx2, ry2 = region["box"]
        if (abs(rx1 - x1) <= slack and abs(ry1 - y1) <= slack
                and abs(rx2 - x2) <= slack and abs(ry2 - y2) <= slack):
            return True
    return False


# ---- the claim: controls, not wallpaper ----

def test_a_wallpaper_on_its_own_has_no_blobs_in_it(tmp_path):
    """A gradient is large, colourful and completely without structure. If it
    scored, every look would spend its passes on the desktop background."""
    im, _, path = _screen(tmp_path)
    assert blobs.salient_regions(_save(im, path)) == []


def test_a_blank_screen_is_empty_rather_than_one_big_region(tmp_path):
    """FIND_EDGES draws a bright frame around any image, which left in would
    make every blank screen report one region the size of itself."""
    from PIL import Image

    path = tmp_path / "blank.png"
    Image.new("RGB", (1280, 800), (252, 252, 252)).save(path)
    assert blobs.salient_regions(str(path)) == []


def test_the_controls_on_a_wallpaper_are_what_comes_back(tmp_path):
    im, draw, path = _screen(tmp_path)
    draw.rectangle([100, 100, 260, 140], fill=(240, 240, 240), outline=(20, 20, 20))
    draw.text((120, 115), "Submit", fill=(0, 0, 0))
    draw.rectangle([400, 300, 760, 340], fill=(255, 255, 255), outline=(120, 120, 120))
    draw.text((410, 315), "type here", fill=(30, 30, 30))

    regions = blobs.salient_regions(_save(im, path))

    assert len(regions) == 2, blobs.format_regions(regions)
    assert _contains(regions, (100, 100, 260, 140))
    assert _contains(regions, (400, 300, 760, 340))


def test_a_centre_is_offered_for_every_region(tmp_path):
    """The coordinate a click needs. It comes from the pixels, not from a
    model — which is the half of grounding the VLM gets wrong."""
    im, draw, path = _screen(tmp_path)
    draw.rectangle([100, 100, 300, 200], fill=(250, 250, 250), outline=(0, 0, 0))

    region = blobs.salient_regions(_save(im, path))[0]
    x1, y1, x2, y2 = region["box"]

    assert region["center"] == ((x1 + x2) // 2, (y1 + y2) // 2)
    assert 180 <= region["center"][0] <= 220
    assert 130 <= region["center"][1] <= 170


def test_the_busier_region_scores_higher(tmp_path):
    """Density and contrast, so a crisp control beats a faint smudge of the
    same size and the budget of crops goes to what is worth naming."""
    im, draw, path = _screen(tmp_path)
    draw.rectangle([100, 100, 300, 160], fill=(255, 255, 255), outline=(0, 0, 0))
    draw.text((110, 125), "Send message", fill=(0, 0, 0))
    draw.rectangle([600, 500, 800, 560], fill=(70, 90, 115), outline=(80, 100, 125))

    regions = blobs.salient_regions(_save(im, path))

    assert regions, "the white control at least must be found"
    assert regions[0]["box"][0] < 400, blobs.format_regions(regions)


# ---- the failure that produced the merge guard ----

def test_a_full_width_bar_does_not_swallow_the_rest_of_the_screen(tmp_path):
    """A menu bar reaches both edges. Merged by proximity without a ceiling,
    everything near it chains in and the answer is one box over the desktop."""
    im, draw, path = _screen(tmp_path)
    draw.rectangle([0, 0, 1199, 28], fill=(245, 245, 245), outline=(180, 180, 180))
    draw.text((10, 8), "File  Edit  View", fill=(0, 0, 0))
    draw.rectangle([500, 400, 700, 450], fill=(255, 255, 255), outline=(0, 0, 0))

    regions = blobs.salient_regions(_save(im, path))
    widest = max(r["box"][2] - r["box"][0] for r in regions)
    tallest = max(r["box"][3] - r["box"][1] for r in regions)

    assert len(regions) >= 2, blobs.format_regions(regions)
    assert tallest < 400, "a region spanning the screen is context, not a target"
    assert widest <= 1200


def test_a_region_covering_most_of_the_screen_is_dropped(tmp_path):
    """A window is where things are, not a thing. Reported as one blob it
    would use the whole crop budget to say "a window"."""
    im, draw, path = _screen(tmp_path)
    draw.rectangle([50, 50, 1150, 750], fill=(250, 250, 250), outline=(0, 0, 0))

    for region in blobs.salient_regions(_save(im, path)):
        box = region["box"]
        area = (box[2] - box[0]) * (box[3] - box[1])
        assert area < 1200 * 800 * 0.5, blobs.format_regions([region])


# ---- the shape of the answer ----

def test_regions_come_back_best_first_and_within_the_limit(tmp_path):
    im, draw, path = _screen(tmp_path)
    for index in range(12):
        x, y = 60 + (index % 4) * 280, 80 + (index // 4) * 220
        draw.rectangle([x, y, x + 180, y + 60], fill=(255, 255, 255), outline=(0, 0, 0))
        draw.text((x + 10, y + 20), f"button {index}", fill=(0, 0, 0))

    regions = blobs.salient_regions(_save(im, path), limit=5)

    assert len(regions) == 5
    assert [r["score"] for r in regions] == sorted(
        (r["score"] for r in regions), reverse=True)


def test_boxes_are_in_original_pixels_not_work_scale(tmp_path):
    """Everything is measured on a downscaled copy; a box that came back in
    that space would be a click at a third of the way up the screen."""
    im, draw, path = _screen(tmp_path, size=(2400, 1600))
    draw.rectangle([1800, 1200, 2100, 1300], fill=(255, 255, 255), outline=(0, 0, 0))

    region = blobs.salient_regions(_save(im, path))[0]

    assert 1750 <= region["box"][0] <= 1850
    assert 1150 <= region["box"][1] <= 1250


def test_formatting_leads_with_the_coordinate(tmp_path):
    im, draw, path = _screen(tmp_path)
    draw.rectangle([100, 100, 300, 160], fill=(255, 255, 255), outline=(0, 0, 0))

    line = blobs.format_regions(blobs.salient_regions(_save(im, path)))

    assert line.strip().startswith("(")


def test_no_regions_says_so_rather_than_returning_nothing():
    assert "No distinct regions" in blobs.format_regions([])


def test_a_screen_is_read_in_well_under_a_second(tmp_path):
    """This runs before every look. A second here is a second added to every
    screenshot the assistant takes."""
    import time

    im, draw, path = _screen(tmp_path, size=(1920, 1080))
    for index in range(30):
        x, y = 40 + (index % 6) * 300, 60 + (index // 6) * 200
        draw.rectangle([x, y, x + 200, y + 70], fill=(255, 255, 255), outline=(0, 0, 0))
    target = _save(im, path)

    start = time.time()
    blobs.salient_regions(target)

    assert time.time() - start < 1.0
