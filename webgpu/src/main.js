async function init() {
  if (!navigator.gpu) {
    document.getElementById('loading').innerHTML = 'WebGPU not supported.';
    return;
  }

  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    document.getElementById('loading').innerHTML = 'No WebGPU adapter found.';
    return;
  }
  const device = await adapter.requestDevice();

  const canvas = document.querySelector('canvas');
  const context = canvas.getContext('webgpu');
  const format = navigator.gpu.getPreferredCanvasFormat();

  context.configure({
    device,
    format,
    alphaMode: 'premultiplied',
  });

  const help = document.getElementById('help');
  const ratio = window.devicePixelRatio || 1;

  function onResize() {
    // Logic from @webgl/main.js: width = innerWidth - help.clientWidth - 20
    const width = window.innerWidth - help.clientWidth - 20;
    const height = window.innerHeight;

    // Set actual canvas size (accounting for pixel ratio)
    canvas.width = Math.floor(width * ratio);
    canvas.height = Math.floor(height * ratio);

    // Set CSS display size
    canvas.style.width = width + 'px';
    canvas.style.height = height + 'px';

    render();
  }

  window.addEventListener('resize', onResize);
  
  // Clear loading message
  document.getElementById('loading').innerHTML = '';
  
  // Initial resize
  onResize();

  function render() {
    // Empty render function
  }

  function animate() {
    requestAnimationFrame(animate);
    render();
  }

  requestAnimationFrame(animate);
}

init();
