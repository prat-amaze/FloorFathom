# Edge Cases

Edge cases present in the capture data that the pipeline must handle.

## 2026-09-19

- **Mirrors:** full-length mirror in Bedroom 2 (`images/b2`, shows a full-body
  reflection) and a small wall mirror over the wash basin in Hall. Reflected space
  must not be reconstructed as real geometry.

- **Reflective floor:** glossy white floor tiles in Hall. Reflections must not be
  read as geometry or damage.

- **Open-plan Hall:** kitchen, living/dining and entry share one space and one photo
  folder. Must produce one room with the zones inside it, not separate rooms.

- **Non-rectangular rooms:** Hall is an irregular quadrilateral and the bathroom is
  L-shaped. Right angles must not be assumed.

- **Hall damage (in the `Hall` set):** a branching hairline crack on the kitchen
  backsplash below the chimney hood (12 × 22 cm), and a staged leakage mark
  (tea-stained paper with drip trails) on the wall beside the bathroom door
  (21 × 25 cm).

- **Bathroom damage:** real ceiling crack. Must be detected and kept separate from
  the Hall damage.

- **Low light:** Hall photos and videos were re-captured in the evening under
  artificial light, with a dark balcony behind the glass door.
