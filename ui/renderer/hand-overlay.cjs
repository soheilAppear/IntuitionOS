// Both camera views use this drawing-only helper; camera ownership stays with their controllers.
const HAND_CONNECTIONS = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],
  [5,9],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15],[15,16],
  [13,17],[0,17],[17,18],[18,19],[19,20]];

function drawHandOverlay(context, snapshot, width, height) {
  // Invalid dimensions cannot turn normalized landmarks into meaningful pixels.
  if (!context || !snapshot || !Number.isFinite(width) || !Number.isFinite(height)
    || width <= 0 || height <= 0) return false;
  const region = snapshot.control_region;
  // The shared pointer region is descriptive; drawing it never enables movement.
  if (snapshot.input_mode === 'mouse' && Array.isArray(region) && region.length === 4
    && region.every(value => Number.isFinite(value) && value >= 0 && value <= 1)
    && region[0] < region[2] && region[1] < region[3]) {
    context.strokeStyle = '#93c5fd'; context.lineWidth = 1; context.setLineDash([5, 5]);
    context.strokeRect(region[0] * width, region[1] * height,
      (region[2] - region[0]) * width, (region[3] - region[1]) * height);
    context.setLineDash([]); context.fillStyle = '#bfdbfe'; context.font = '12px sans-serif';
    context.fillText('Pointer area', region[0] * width + 6, region[1] * height + 16);
  }
  const points = snapshot.landmarks;
  // A stale/untracked set must not leave a plausible hand on a current image.
  if (snapshot.tracked !== true || !Array.isArray(points) || points.length !== 21
    || !points.every(point => Array.isArray(point) && Number.isFinite(point[0]) && Number.isFinite(point[1]))) return false;
  context.strokeStyle = '#5eead4'; context.lineWidth = 2; context.beginPath();
  for (const [a, b] of HAND_CONNECTIONS) {
    context.moveTo(points[a][0] * width, points[a][1] * height);
    context.lineTo(points[b][0] * width, points[b][1] * height);
  }
  context.stroke(); context.fillStyle = '#fef3c7';
  // Draw all 21 joint markers, including fingertips and the wrist, in both views.
  for (const point of points) {
    context.beginPath(); context.arc(point[0] * width, point[1] * height, 3, 0, Math.PI * 2); context.fill();
  }
  return true;
}

module.exports = { drawHandOverlay };
