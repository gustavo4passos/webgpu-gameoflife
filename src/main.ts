import * as Grid from './grid.ts'

if (!navigator.gpu) {
    throw new Error("WebGPU not available.")
} 

const adapter = await navigator.gpu.requestAdapter()

if (!adapter) {
    throw new Error("No appropriate GPUAdapter found.")
}

const device = await adapter.requestDevice()

const canvas = document.querySelector<HTMLCanvasElement>("#canvas")!
const windowSide = Math.min(window.innerWidth, window.innerHeight)
canvas.width = windowSide;
canvas.height = windowSide;
const GRID_SIZE = Math.floor(windowSide / 10);

const context = canvas.getContext('webgpu')

if (!context) {
    throw new Error("Unable to get canvas context")
}

const canvasFormat = navigator.gpu.getPreferredCanvasFormat()

context.configure({
    device: device,
    format: canvasFormat
})

const vertices = new Float32Array([
  -0.8, -0.8,
   0.8, -0.8,
   0.8,  0.8,

  -0.8, -0.8,
   0.8,  0.8,
  -0.8,  0.8
])

const vertexBuffer = device.createBuffer({
    label: "Cell Vertices",
    size: vertices.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
})

device.queue.writeBuffer(vertexBuffer, 0, vertices)

const uniformArray = new Float32Array([GRID_SIZE, GRID_SIZE])
const uniformBuffer = device.createBuffer({
  label: "Grid Uniforms",
  size: uniformArray.byteLength,
  usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
})

device.queue.writeBuffer(uniformBuffer, 0, uniformArray)

const cellStateArray = new Uint32Array(GRID_SIZE * GRID_SIZE)

const storageBuffers = [
    device.createBuffer({
    label: "Cell State 1",
    size: cellStateArray.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    }),
    device.createBuffer({
        label: "Cell State 2",
        size: cellStateArray.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    })
]

for (let i = 0; i < cellStateArray.length; i++) {
  if (Math.random() < 0.5) cellStateArray[i] = 1;
  else cellStateArray[i] = 0;
}

device.queue.writeBuffer(storageBuffers[0], 0, cellStateArray)
device.queue.writeBuffer(storageBuffers[1], 0, cellStateArray)

const vertexBufferLayout: GPUVertexBufferLayout = {
    arrayStride: 8,
    attributes: [{
        format: "float32x2",
        offset: 0,
        shaderLocation: 0
    }]
}

const cellShaderModule = device.createShaderModule({
    label: "Cell Shader",
    code: `
        @group(0) @binding(0) var<uniform> grid: vec2f;
        @group(0) @binding(1) var<storage> cellState: array<u32>;

        struct VertexInput {
            @location(0) pos: vec2f,
            @builtin(instance_index) instanceIdx: u32
        };

        struct VertexOutput {
            @builtin(position) pos: vec4f,
            @location(0) cell: vec2f,
        };

        @vertex
        fn vertexMain(input: VertexInput) -> VertexOutput {
            let i = f32(input.instanceIdx);
            let state = cellState[input.instanceIdx];

            let cell = vec2f(i % grid.x, floor(i / grid.y));
            let cellOffset = cell / grid * 2;
            let finalPos = (input.pos * f32(state) + 1) / grid - 1 + cellOffset;

            var output: VertexOutput;
            output.pos = vec4(finalPos, 0, 1);
            output.cell = cell;
            return output;
        }

        @fragment
        fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
            let cellNormalized = input.cell /grid;
            return vec4f(cellNormalized, 1 - cellNormalized.x - 0.2, 1);
        }
    `
})

const  WORKGROUP_SIZE = 8;
const simulationShaderModule = device.createShaderModule({
  label: "Simulation Shader",
  code: `
    @group(0) @binding(0) var<uniform> grid: vec2f;
    @group(0) @binding(1) var<storage>             cellStateIn:  array<u32>;
    @group(0) @binding(2) var<storage, read_write> cellStateOut: array<u32>;

    fn getCellIndex(cell: vec2i) -> i32 {
      let iGrid = vec2i(grid);
      return (cell.y % iGrid.y) * iGrid.x + (cell.x % iGrid.x);
    }

    fn isCellActive(x: i32, y: i32) -> u32 {
      return cellStateIn[getCellIndex(vec2i(x, y))];
    }

    @compute @workgroup_size(${WORKGROUP_SIZE}, ${WORKGROUP_SIZE})
    fn computeMain(@builtin(global_invocation_id) cell: vec3u) {
      let cellIdx = getCellIndex(vec2i(cell.xy));

      let iCell = vec2i(cell.xy);
      let nActiveNeighbors = isCellActive(iCell.x - 1, iCell.y    ) +
                       isCellActive(iCell.x - 1, iCell.y + 1) + 
                       isCellActive(iCell.x    , iCell.y + 1) + 
                       isCellActive(iCell.x + 1, iCell.y + 1) + 
                       isCellActive(iCell.x + 1, iCell.y    ) + 
                       isCellActive(iCell.x + 1, iCell.y - 1) + 
                       isCellActive(iCell.x    , iCell.y - 1) + 
                       isCellActive(iCell.x - 1, iCell.y - 1);

      switch nActiveNeighbors {
        case 2: {
          cellStateOut[cellIdx] = cellStateIn[cellIdx];
        }
        case 3: {
          cellStateOut[cellIdx] = 1;
        }
        default: {
          cellStateOut[cellIdx] = 0;
        }
      }
    }
  `
})


const bindGroupLayout = device.createBindGroupLayout({
  label: "Cell Bind Group Layout",
  entries: [
    {
      binding: 0,
      visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT | GPUShaderStage.COMPUTE,
      buffer: { type: 'uniform' }
    },
    {
      binding: 1,
      visibility: GPUShaderStage.VERTEX | GPUShaderStage.COMPUTE,
      buffer: { type: 'read-only-storage' }
    },
    {
      binding: 2,
      visibility: GPUShaderStage.COMPUTE,
      buffer: { type: 'storage' }
    },
  ]
})

const bindGroups = [
    device.createBindGroup({
        label: "Cell renderer bind group",
        layout: bindGroupLayout,
        entries: [{
            binding: 0,
            resource: { buffer: uniformBuffer }
        },
        {
            binding: 1,
            resource: { buffer: storageBuffers[0] }
        },
        {
            binding: 2,
            resource: { buffer: storageBuffers[1] }
        }],
    }),
    device.createBindGroup({
        label: "Cell renderer bind group 1",
        layout: bindGroupLayout,
        entries: [{
            binding: 0,
            resource: { buffer: uniformBuffer } 
        },
        {
            binding: 1,
            resource: { buffer: storageBuffers[1] },
        },
        {
          binding: 2,
          resource: { buffer: storageBuffers[0] },
        }],
    })
]

const pipelineLayout = device.createPipelineLayout({
  label: "Cell Pipeline Layout",
  bindGroupLayouts: [ bindGroupLayout ]
})

const cellPipeline = device.createRenderPipeline({
    label: "Cell Pipeline",
    layout: pipelineLayout,
    vertex: {
        module: cellShaderModule,
        entryPoint: "vertexMain",
        buffers: [vertexBufferLayout],
    },
    fragment: {
        module: cellShaderModule,
        entryPoint: "fragmentMain",
        targets: [{
            format: canvasFormat
        }]
    }
})

const computePipeline = device.createComputePipeline({
  label: "Compute pipeline",
  layout: pipelineLayout,
  compute: {
    module: simulationShaderModule,
    entryPoint: "computeMain",
  }
})

const UPDATE_INTERVAL_MS = 200;
let step = 0;

function update() {
    const encoder = device.createCommandEncoder()

    const computePass = encoder.beginComputePass()
    computePass.setPipeline(computePipeline)
    computePass.setBindGroup(0, bindGroups[step % 2])
    const workgroupCount = Math.ceil(GRID_SIZE / WORKGROUP_SIZE)
    computePass.dispatchWorkgroups(workgroupCount, workgroupCount)
    computePass.end()

    step += 1;

    const pass = encoder.beginRenderPass({
        colorAttachments: [{
            view: context!.getCurrentTexture().createView(),
            loadOp: "clear",
            storeOp: "store",
            clearValue: { r: 0.15, g: 0.15, b: 0.25, a: 1 }
        }]
    })

    pass.setPipeline(cellPipeline)
    pass.setVertexBuffer(0, vertexBuffer)
    pass.setBindGroup(0, bindGroups[step % 2])
    pass.draw(vertices.length / 2, GRID_SIZE * GRID_SIZE)

    pass.end()

    device.queue.submit([encoder.finish()])
}
        
setInterval(update, UPDATE_INTERVAL_MS)