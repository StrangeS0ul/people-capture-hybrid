/* app.js — People Capture Hybrid (browser, perf-tuned)
   YOLOv8 ONNX + MediaPipe Pose. Client-side fusion with padding-aware mapping,
   UI toggles, and frame-skipping to reduce stutter.
*/

/* global ort, Pose, drawConnectors, drawLandmarks */
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const statusEl = document.getElementById('status');
const confSlider = document.getElementById('conf');
const iouSlider = document.getElementById('iou');

// Inject extra controls without editing HTML
const controls = document.querySelector('.controls') || document.body;
controls.insertAdjacentHTML('beforeend', `
  <label><input id="use-yolo" type="checkbox" checked> Use YOLO</label>
  <label><input id="use-pose" type="checkbox" checked> Use Pose</label>
  <label>YOLO every
    <select id="yolo-every">
      <option value="1">1</option>
      <option value="2" selected>2</option>
      <option value="3">3</option>
      <option value="4">4</option>
    </select> frame(s)
  </label>
  <label>Pose every
    <select id="pose-every">
      <option value="1">1</option>
      <option value="2" selected>2</option>
      <option value="3">3</option>
    </select> frame(s)
  </label>
  <label>TopK <input id="topk" type="range" min="10" max="200" step="10" value="50"></label>
  <label><input id="draw-box" type="checkbox" checked> Draw boxes</label>
  <label><input id="draw-pose" type="checkbox" checked> Draw pose</label>
`);

const INPUT_SIZE = 640;
const MODEL_URL = './models/yolov8n.onnx';

let session, pose;
let frameId = 0;
let yoloBusy = false, poseBusy = false;
let lastDetections = [];   // [{box:[x1,y1,x2,y2], score:float}]
let lastPoseRes = null;

// ---------- UI getters
function ui() {
  return {
    conf: parseFloat(confSlider.value),
    iou: parseFloat(iouSlider.value),
    useYolo: document.getElementById('use-yolo').checked,
    usePose: document.getElementById('use-pose').checked,
    yoloEvery: parseInt(document.getElementById('yolo-every').value, 10),
    poseEvery: parseInt(document.getElementById('pose-every').value, 10),
    topk: parseInt(document.getElementById('topk').value, 10),
    drawBox: document.getElementById('draw-box').checked,
    drawPose: document.getElementById('draw-pose').checked,
  };
}

// ---------- Helpers
const sigmoid = (x)=> 1/(1+Math.exp(-x));

function letterbox(source, size) {
  const srcW = source.videoWidth || source.width;
  const srcH = source.videoHeight || source.height;
  const scale = Math.min(size/srcW, size/srcH);
  const newW = Math.round(srcW * scale);
  const newH = Math.round(srcH * scale);
  const dx = Math.floor((size - newW)/2);
  const dy = Math.floor((size - newH)/2);
  const tmp = document.createElement('canvas');
  tmp.width = size; tmp.height = size;
  const tctx = tmp.getContext('2d');
  tctx.fillStyle = '#000'; tctx.fillRect(0,0,size,size);
  tctx.drawImage(source, 0, 0, srcW, srcH, dx, dy, newW, newH);
  return {canvas: tmp, scale, dx, dy, newW, newH};
}

function xywh2xyxy(x,y,w,h){ return [x-w/2, y-h/2, x+w/2, y+h/2]; }

function boxIoU(a,b){
  const [ax1,ay1,ax2,ay2] = a, [bx1,by1,bx2,by2] = b;
  const ix1=Math.max(ax1,bx1), iy1=Math.max(ay1,by1);
  const ix2=Math.min(ax2,bx2), iy2=Math.min(ay2,by2);
  const inter = Math.max(0, ix2-ix1)*Math.max(0, iy2-iy1);
  const areaA = Math.max(0, ax2-ax1)*Math.max(0, ay2-ay1);
  const areaB = Math.max(0, bx2-bx1)*Math.max(0, by2-by1);
  const union = areaA+areaB-inter;
  return union ? inter/union : 0;
}

function nms(boxes, scores, iouTh){
  const idxs = scores.map((s,i)=>[s,i]).sort((a,b)=>b[0]-a[0]).map(x=>x[1]);
  const keep=[];
  while (idxs.length){
    const cur = idxs.shift(); keep.push(cur);
    for (let i=idxs.length-1;i>=0;i--){
      if (boxIoU(boxes[cur], boxes[idxs[i]]) > iouTh) idxs.splice(i,1);
    }
  }
  return keep;
}

function landmarkInBox(lm, box, imgW, imgH){
  const [x1,y1,x2,y2] = box;
  const x = lm.x * imgW, y = lm.y * imgH;
  return x>=x1 && x<=x2 && y>=y1 && y<=y2;
}

function fuseDetections(yoloDetections, poseResult, imgW, imgH){
  const lms = poseResult?.poseLandmarks || [];
  const visible = lms.filter(l=> (l.visibility??0) > 0.5);
  const poseScore = lms.length ? visible.length / lms.length : 0;
  return yoloDetections.map(det=>{
    const inside = visible.filter(lm=> landmarkInBox(lm, det.box, imgW, imgH)).length;
    const confirmed = inside >= 8;
    const fusedScore = confirmed ? (0.7*det.score + 0.3*poseScore) : (0.5*det.score);
    return {...det, fusedScore, confirmed};
  });
}

// Map letterboxed coords back to original video using padding info
function scaleBoxesToVideo(boxes, vidW, vidH, dx, dy, scale){
  return boxes.map(([x1,y1,x2,y2])=>{
    const ux1 = (x1 - dx) / scale;
    const uy1 = (y1 - dy) / scale;
    const ux2 = (x2 - dx) / scale;
    const uy2 = (y2 - dy) / scale;
    const X1 = Math.max(0, Math.min(vidW, ux1));
    const Y1 = Math.max(0, Math.min(vidH, uy1));
    const X2 = Math.max(0, Math.min(vidW, ux2));
    const Y2 = Math.max(0, Math.min(vidH, uy2));
    return [X1, Y1, X2, Y2];
  });
}

function drawFused(ctx, fused){
  fused.forEach(d=>{
    const [x1,y1,x2,y2] = d.box;
    ctx.strokeStyle = d.confirmed ? '#00d1ff' : '#ffaa00';
    ctx.lineWidth = 2;
    ctx.strokeRect(x1,y1, x2-x1, y2-y1);
    const label = (d.confirmed?'CONF ':'') + (d.fusedScore??d.score).toFixed(2);
    ctx.fillStyle = 'rgba(0,0,0,0.6)';
    const tw = ctx.measureText(label).width+10;
    ctx.fillRect(x1, Math.max(0,y1-16), tw, 16);
    ctx.fillStyle = '#eaeaea';
    ctx.fillText(label, x1+4, Math.max(10,y1-4));
  });
}

// ---------- Setup
async function setup(){
  statusEl.textContent = 'Loading model…';
  try {
    session = await ort.InferenceSession.create(MODEL_URL, {
      executionProviders: ['webgl','wasm'],
      graphOptimizationLevel: 'all'
    });
  } catch (e) {
    console.warn('Falling back to WASM only:', e);
    session = await ort.InferenceSession.create(MODEL_URL, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all'
    });
  }

  statusEl.textContent = 'Loading MediaPipe Pose…';
  pose = new Pose({
    locateFile: (file)=>`https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`
  });
  // cache latest results without blocking the render loop
  pose.onResults((r)=>{ lastPoseRes = r; });
  pose.setOptions({ modelComplexity:1, smoothLandmarks:true, minDetectionConfidence:0.5, minTrackingConfidence:0.5 });

  const stream = await navigator.mediaDevices.getUserMedia({video:{facingMode:'user', width:{ideal:1280}, height:{ideal:720}}, audio:false});
  video.srcObject = stream;
  await video.play();
  canvas.width = video.videoWidth; canvas.height = video.videoHeight;

  statusEl.textContent = 'Running…';
  requestAnimationFrame(loop);
}

// ---------- Decode YOLOv8 ONNX output (supports [1,84,8400] or [1,8400,84])
function decodeYOLOv8(out, confTh){
  const {data, dims} = out;
  let boxes=[], scores=[];
  if (dims.length !== 3) return {boxes, scores};
  const A = dims[1], B = dims[2];

  function pushBox(x,y,w,h,cls0){
    // if normalized, convert to px
    if (w <= 1.0 && h <= 1.0){
      x *= INPUT_SIZE; y *= INPUT_SIZE; w *= INPUT_SIZE; h *= INPUT_SIZE;
    }
    if (cls0 >= confTh){
      boxes.push(xywh2xyxy(x,y,w,h)); scores.push(cls0);
    }
  }

  if (A === 84){ // [1,84,8400]
    for (let k=0;k<B;k++){
      const x = data[0*A*B + 0*B + k];
      const y = data[0*A*B + 1*B + k];
      const w = data[0*A*B + 2*B + k];
      const h = data[0*A*B + 3*B + k];
      const cls0 = sigmoid(data[0*A*B + (4+0)*B + k]); // person only
      pushBox(x,y,w,h,cls0);
    }
  } else if (B === 84){ // [1,8400,84]
    for (let i=0;i<A;i++){
      const base = i*84;
      const x = data[base+0], y=data[base+1], w=data[base+2], h=data[base+3];
      const cls0 = sigmoid(data[base+4+0]);
      pushBox(x,y,w,h,cls0);
    }
  }
  return {boxes, scores};
}

// ---------- Main loop with frame-skipping and cached results
async function loop(){
  const {
    conf, iou, useYolo, usePose, yoloEvery, poseEvery, topk, drawBox, drawPose
  } = ui();

  // Preprocess once per frame for both Pose and YOLO
  const {canvas: lb, dx, dy, scale} = letterbox(video, INPUT_SIZE);
  const imgData = lb.getContext('2d').getImageData(0,0,INPUT_SIZE,INPUT_SIZE).data;

  // Run YOLO every N frames, if enabled and not busy
  if (useYolo && !yoloBusy && (frameId % yoloEvery === 0)) {
    yoloBusy = true;
    // NHWC -> NCHW float32
    const chw = new Float32Array(1*3*INPUT_SIZE*INPUT_SIZE);
    let outIdx = 0;
    for (let c=0;c<3;c++){
      for (let y=0;y<INPUT_SIZE;y++){
        for (let x=0;x<INPUT_SIZE;x++){
          const idx = (y*INPUT_SIZE + x)*4 + c;
          chw[outIdx++] = imgData[idx]/255;
        }
      }
    }
    const tensor = new ort.Tensor('float32', chw, [1,3,INPUT_SIZE,INPUT_SIZE]);
    const yoloOut = await session.run({images: tensor}).catch(()=>session.run({input: tensor}));
    const outName = Object.keys(yoloOut)[0];
    const out = yoloOut[outName];
    let {boxes, scores} = decodeYOLOv8(out, conf);
    // TopK thinning
    const order = scores.map((s,i)=>[s,i]).sort((a,b)=>b[0]-a[0]).slice(0,topk).map(x=>x[1]);
    boxes = order.map(i=>boxes[i]); scores = order.map(i=>scores[i]);
    const boxesScaled = scaleBoxesToVideo(boxes, video.videoWidth, video.videoHeight, dx, dy, scale);
    const keepIdx = nms(boxesScaled, scores, iou);
    lastDetections = keepIdx.map(i=>({box: boxesScaled[i], score: scores[i]}));
    yoloBusy = false;
  }

  // Run Pose every M frames, if enabled and not busy
  if (usePose && !poseBusy && (frameId % poseEvery === 0)) {
    poseBusy = true;
    // pose.send uses the HTMLVideoElement directly
    pose.send({image: video}).then(()=>{ poseBusy = false; }).catch(()=>{ poseBusy = false; });
  }

  // Draw current frame
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  if (drawPose && lastPoseRes?.poseLandmarks){
    drawConnectors(ctx, lastPoseRes.poseLandmarks, Pose.POSE_CONNECTIONS, {color:'#7af', lineWidth:2});
    drawLandmarks(ctx, lastPoseRes.poseLandmarks, {color:'#fff', lineWidth:1});
  }
  let toDraw = lastDetections;
  if (useYolo && usePose && lastPoseRes){
    toDraw = fuseDetections(lastDetections, lastPoseRes, video.videoWidth, video.videoHeight);
  }
  if (drawBox && toDraw?.length){
    drawFused(ctx, toDraw);
  }

  frameId++;
  requestAnimationFrame(loop);
}

// Start
setup().catch(e=>{ console.error(e); statusEl.textContent = 'Error: ' + e.message; });
