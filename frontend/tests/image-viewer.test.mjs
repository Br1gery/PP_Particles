import test from "node:test";
import assert from "node:assert/strict";
import { fitSize, constrainOffset, zoomAt, detectionAt, imagePoint, ImageViewer } from "../image-viewer.mjs";

function interactiveViewer(t, callbacks={}) {
  const oldWindow=globalThis.window, oldResizeObserver=globalThis.ResizeObserver;
  globalThis.window={devicePixelRatio:1};globalThis.ResizeObserver=class { observe() {} };
  t.after(()=>{globalThis.window=oldWindow;globalThis.ResizeObserver=oldResizeObserver;});
  const target=()=>({events:{},style:{},classList:{toggle(){},add(){},remove(){}},addEventListener(name,callback){this.events[name]=callback;}});
  const stage=Object.assign(target(),{clientWidth:100,clientHeight:100,getBoundingClientRect:()=>({left:0,top:0}),
    setPointerCapture(){},releasePointerCapture(){},hasPointerCapture:()=>false});
  const context={setTransform(){},clearRect(){},save(){},restore(){},setLineDash(){},strokeRect(){},
    beginPath(){},lineTo(){},moveTo(){},stroke(){},arc(){},fill(){},ellipse(){this.ellipses=(this.ellipses||0)+1;}};
  const viewer=new ImageViewer({stage,image:target(),controls:target(),zoomIn:target(),zoomOut:target(),reset:target(),value:target(),
    tooltip:Object.assign(target(),{dataset:{}}),overlay:{width:0,height:0,getContext:()=>context},...callbacks});
  Object.assign(viewer,{loaded:true,imageWidth:200,imageHeight:100,baseSize:{width:100,height:50},view:{scale:2,x:-20,y:0}});
  const fire=(name,x,y,extra={})=>stage.events[name]({button:0,isPrimary:true,pointerId:1,clientX:x,clientY:y,preventDefault(){},...extra});
  return {viewer,fire,context};
}

test("ellipse gestures use source coordinates after zoom and never select a particle", t=>{
  let box,selected=false;
  const {viewer,fire,context}=interactiveViewer(t,{onDraw:value=>box=value,onSelect:()=>selected=true});
  viewer.setDrawing(true,"ellipse");
  fire("pointerdown",10,10);fire("pointermove",60,40);fire("pointerup",60,40);
  assert.deepEqual(box,[30,10,80,40]);assert.equal(selected,false);assert.ok(context.ellipses>0);
});

test("polygon gestures support finishing, point undo, pan and cancellation", t=>{
  let polygon,cancelled=false;
  const {viewer,fire}=interactiveViewer(t,{onPolygon:value=>polygon=value,onDrawCancel:()=>cancelled=true});
  viewer.setDrawing(true,"polygon");
  for (const [x,y] of [[10,10],[60,10],[60,40]]) {fire("pointerdown",x,y);fire("pointerup",x,y);}
  fire("keydown",0,0,{key:"Enter"});
  assert.deepEqual(polygon,[[30,10],[80,10],[80,40]]);
  fire("keydown",0,0,{key:"Backspace"});assert.equal(viewer.polygonPoints.length,2);
  fire("pointerdown",50,50,{shiftKey:true});fire("pointermove",30,50);fire("pointerup",30,50);
  assert.equal(viewer.polygonPoints.length,2);assert.equal(viewer.view.x,-40);
  fire("keydown",0,0,{key:"Escape"});
  assert.equal(cancelled,true);assert.equal(viewer.drawing,false);assert.equal(viewer.polygonPoints.length,0);
});

test("fit keeps the whole image visible without distortion or upsampling", () => {
  assert.deepEqual(fitSize(2000, 1000, 1000, 600), { width: 1000, height: 500 });
  assert.deepEqual(fitSize(1000, 2000, 1000, 600), { width: 300, height: 600 });
  assert.deepEqual(fitSize(100, 100, 1000, 600), { width: 100, height: 100 });
  assert.deepEqual(fitSize(100, 100, 0, 0), { width: 0, height: 0 });
});

test("zoom preserves the image point under the mouse cursor", () => {
  const image = { width: 1000, height: 500 };
  const viewport = { width: 1000, height: 600 };
  const before = { scale: 1, x: 0, y: 50 };
  const anchor = { x: 250, y: 200 };
  const after = zoomAt(before, 2, anchor, image, viewport);
  assert.equal((anchor.x - before.x) / before.scale, (anchor.x - after.x) / after.scale);
  assert.equal((anchor.y - before.y) / before.scale, (anchor.y - after.y) / after.scale);
});

test("panning stays within the image and centers smaller dimensions", () => {
  assert.equal(constrainOffset(100, 2000, 1000), 0);
  assert.equal(constrainOffset(-1200, 2000, 1000), -1000);
  assert.equal(constrainOffset(-250, 2000, 1000), -250);
  assert.equal(constrainOffset(-250, 300, 600), 150);
});

test("zoom limits and reset bring the complete image back into view", () => {
  const image = { width: 1000, height: 500 };
  const viewport = { width: 1000, height: 600 };
  const anchor = { x: 500, y: 300 };
  const enlarged = zoomAt({ scale: 1, x: 0, y: 50 }, 100, anchor, image, viewport);
  assert.equal(enlarged.scale, 8);
  const reset = zoomAt(enlarged, .1, anchor, image, viewport);
  assert.deepEqual(reset, { scale: 1, x: 0, y: 50 });
});

test("hover finds particles in original coordinates after zooming and panning", () => {
  const particle = { id: 7, bbox: [450, 250, 550, 350] };
  const displayed = { width: 1000, height: 500 };
  const original = { width: 2000, height: 1000 };
  const view = { scale: 2, x: -400, y: -200 };
  assert.equal(detectionAt({ x: 100, y: 100 }, [particle], view, displayed, original), particle);
  assert.equal(detectionAt({ x: 400, y: 100 }, [particle], view, displayed, original), null);
});

test("hover ignores empty margins and prefers the smaller overlapping particle", () => {
  const large = { id: 1, bbox: [0, 0, 1000, 1000] };
  const small = { id: 2, bbox: [100, 100, 200, 200] };
  const view = { scale: 1, x: 250, y: 0 };
  const displayed = { width: 500, height: 500 };
  const original = { width: 1000, height: 1000 };
  assert.equal(detectionAt({ x: 325, y: 75 }, [large, small], view, displayed, original), small);
  assert.equal(detectionAt({ x: 10, y: 10 }, [large, small], view, displayed, original), null);
  assert.equal(detectionAt({ x: 325, y: 75 }, [], view, displayed, original), null);
});

test("manual drawing maps screen coordinates back to the source after zoom and pan", () => {
  const view = { scale: 2, x: -400, y: -200 };
  const displayed = { width: 1000, height: 500 };
  const original = { width: 2000, height: 1000 };
  assert.deepEqual(imagePoint({ x: 100, y: 100 }, view, displayed, original), { x: 500, y: 300 });
  assert.deepEqual(imagePoint({ x: -1000, y: 2000 }, view, displayed, original), { x: 0, y: 1000 });
});
