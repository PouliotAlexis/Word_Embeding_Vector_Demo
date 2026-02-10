/**
 * Word Embedding Explorer - Visualizations
 * D3.js and Three.js visualizations
 */

// ============================================
// Neighbors Visualization (D3.js)
// ============================================

function renderNeighborsViz(points, centerWord, neighbors) {
    const container = document.getElementById('neighbors-viz');
    container.innerHTML = '';

    const width = container.clientWidth || 600;
    const height = 400;

    // Create SVG
    const svg = d3.select('#neighbors-viz')
        .append('svg')
        .attr('width', width)
        .attr('height', height);

    // Add gradient definitions
    const defs = svg.append('defs');

    const gradient = defs.append('radialGradient')
        .attr('id', 'nodeGradient');

    gradient.append('stop')
        .attr('offset', '0%')
        .attr('stop-color', '#8b5cf6');

    gradient.append('stop')
        .attr('offset', '100%')
        .attr('stop-color', '#6366f1');

    // Calculate scales
    const xExtent = d3.extent(points, d => d.x);
    const yExtent = d3.extent(points, d => d.y);

    const xScale = d3.scaleLinear()
        .domain(xExtent)
        .range([60, width - 60]);

    const yScale = d3.scaleLinear()
        .domain(yExtent)
        .range([60, height - 60]);

    // Create distance lookup
    const distanceMap = new Map();
    neighbors.forEach(n => distanceMap.set(n.word, n.distance));

    // Draw connecting lines
    const centerPoint = points.find(p => p.word === centerWord);

    if (centerPoint) {
        points.forEach(point => {
            if (point.word !== centerWord) {
                svg.append('line')
                    .attr('x1', xScale(centerPoint.x))
                    .attr('y1', yScale(centerPoint.y))
                    .attr('x2', xScale(point.x))
                    .attr('y2', yScale(point.y))
                    .attr('stroke', '#6366f1')
                    .attr('stroke-opacity', 0.1)
                    .attr('stroke-width', 1);
            }
        });
    }

    // Draw nodes
    const nodes = svg.selectAll('.neighbor-node')
        .data(points)
        .enter()
        .append('g')
        .attr('class', 'neighbor-node')
        .attr('transform', d => `translate(${xScale(d.x)}, ${yScale(d.y)})`)
        .style('cursor', 'pointer')
        .on('mouseover', function (event, d) {
            d3.select(this).select('circle')
                .transition()
                .duration(200)
                .attr('r', d.word === centerWord ? 18 : 14);
        })
        .on('mouseout', function (event, d) {
            d3.select(this).select('circle')
                .transition()
                .duration(200)
                .attr('r', d.word === centerWord ? 15 : 10);
        });

    // Node circles
    nodes.append('circle')
        .attr('r', 0)
        .attr('fill', d => d.word === centerWord ? 'url(#nodeGradient)' : '#6366f1')
        .attr('fill-opacity', d => {
            if (d.word === centerWord) return 1;
            const dist = distanceMap.get(d.word) || 0.5;
            return 1 - dist;
        })
        .attr('stroke', '#fff')
        .attr('stroke-width', d => d.word === centerWord ? 3 : 1)
        .attr('stroke-opacity', 0.3)
        .transition()
        .duration(500)
        .delay((d, i) => i * 20)
        .attr('r', d => d.word === centerWord ? 15 : 10);

    // Node labels
    nodes.append('text')
        .attr('dy', d => d.word === centerWord ? -22 : -15)
        .attr('text-anchor', 'middle')
        .attr('fill', '#fff')
        .attr('font-size', d => d.word === centerWord ? '14px' : '11px')
        .attr('font-weight', d => d.word === centerWord ? '600' : '400')
        .attr('opacity', 0)
        .text(d => d.word)
        .transition()
        .duration(500)
        .delay((d, i) => i * 20 + 200)
        .attr('opacity', 1);
}

// ============================================
// Graph Visualization (D3.js Force-Directed)
// ============================================

function renderGraph(graphData) {
    const container = document.getElementById('graph-svg');
    const width = container.clientWidth || 800;
    const height = 500;

    // Clear previous
    d3.select('#graph-svg').selectAll('*').remove();

    const svg = d3.select('#graph-svg')
        .attr('viewBox', [0, 0, width, height]);

    // Add gradient
    const defs = svg.append('defs');

    const gradient = defs.append('linearGradient')
        .attr('id', 'linkGradient')
        .attr('gradientUnits', 'userSpaceOnUse');

    gradient.append('stop')
        .attr('offset', '0%')
        .attr('stop-color', '#6366f1');

    gradient.append('stop')
        .attr('offset', '100%')
        .attr('stop-color', '#ec4899');

    // Prepare data
    const nodes = graphData.nodes.map(n => ({ ...n }));
    const links = graphData.edges.map(e => ({
        source: e.source,
        target: e.target,
        weight: e.weight
    }));

    // Color scale for groups
    const colorScale = d3.scaleOrdinal()
        .domain([-1, 0, 1, 2, 3, 4])
        .range(['#64748b', '#f43f5e', '#8b5cf6', '#06b6d4', '#10b981', '#f59e0b']);

    // Force simulation
    const simulation = d3.forceSimulation(nodes)
        .force('link', d3.forceLink(links).id(d => d.id).distance(100).strength(0.5))
        .force('charge', d3.forceManyBody().strength(-300))
        .force('center', d3.forceCenter(width / 2, height / 2))
        .force('collision', d3.forceCollide().radius(40));

    // Draw links
    const link = svg.append('g')
        .selectAll('line')
        .data(links)
        .enter()
        .append('line')
        .attr('class', 'graph-link')
        .attr('stroke-width', d => Math.max(1, d.weight * 3))
        .attr('stroke', 'url(#linkGradient)')
        .attr('stroke-opacity', d => 0.2 + d.weight * 0.5);

    // Draw nodes
    const node = svg.append('g')
        .selectAll('.node-group')
        .data(nodes)
        .enter()
        .append('g')
        .attr('class', 'node-group')
        .call(d3.drag()
            .on('start', dragstarted)
            .on('drag', dragged)
            .on('end', dragended));

    // Node circles
    node.append('circle')
        .attr('class', 'graph-node')
        .attr('r', d => d.original ? 20 : 12)
        .attr('fill', d => colorScale(d.group))
        .attr('stroke', '#fff')
        .attr('stroke-width', d => d.original ? 3 : 1)
        .attr('stroke-opacity', 0.5);

    // Glow effect for original nodes
    node.filter(d => d.original)
        .append('circle')
        .attr('r', 25)
        .attr('fill', 'none')
        .attr('stroke', d => colorScale(d.group))
        .attr('stroke-width', 2)
        .attr('stroke-opacity', 0.3)
        .attr('filter', 'blur(3px)');

    // Node labels
    node.append('text')
        .attr('class', 'graph-node-label')
        .attr('dy', d => d.original ? 35 : 25)
        .attr('text-anchor', 'middle')
        .attr('font-size', d => d.original ? '13px' : '10px')
        .attr('font-weight', d => d.original ? '600' : '400')
        .text(d => d.id);

    // Update positions
    simulation.on('tick', () => {
        link
            .attr('x1', d => d.source.x)
            .attr('y1', d => d.source.y)
            .attr('x2', d => d.target.x)
            .attr('y2', d => d.target.y);

        node.attr('transform', d => `translate(${d.x}, ${d.y})`);
    });

    // Drag functions
    function dragstarted(event, d) {
        if (!event.active) simulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
    }

    function dragged(event, d) {
        d.fx = event.x;
        d.fy = event.y;
    }

    function dragended(event, d) {
        if (!event.active) simulation.alphaTarget(0);
        d.fx = null;
        d.fy = null;
    }
}

// ============================================
// 3D Word Cloud (Three.js)
// ============================================

let cloudScene, cloudCamera, cloudRenderer, cloudControls;
let cloudAnimationId;

function render3DCloud(points, seedWords) {
    const container = document.getElementById('cloud-container');

    // Clean up previous
    if (cloudRenderer) {
        cancelAnimationFrame(cloudAnimationId);
        container.innerHTML = '';
    }

    const width = container.clientWidth || 600;
    const height = 450;

    // Scene setup
    cloudScene = new THREE.Scene();
    cloudScene.background = new THREE.Color(0x12121a);

    // Camera
    cloudCamera = new THREE.PerspectiveCamera(75, width / height, 0.1, 1000);
    cloudCamera.position.z = 30;

    // Renderer
    cloudRenderer = new THREE.WebGLRenderer({ antialias: true });
    cloudRenderer.setSize(width, height);
    cloudRenderer.setPixelRatio(window.devicePixelRatio);
    container.appendChild(cloudRenderer.domElement);

    // Controls
    cloudControls = new THREE.OrbitControls(cloudCamera, cloudRenderer.domElement);
    cloudControls.enableDamping = true;
    cloudControls.dampingFactor = 0.05;

    // Normalize coordinates
    const xExtent = d3.extent(points, d => d.x);
    const yExtent = d3.extent(points, d => d.y);
    const zExtent = d3.extent(points, d => d.z);

    const scale = 20;
    const normalize = (val, extent) => {
        const range = extent[1] - extent[0];
        return range === 0 ? 0 : ((val - extent[0]) / range - 0.5) * scale;
    };

    // Colors
    const seedColor = new THREE.Color(0x8b5cf6);
    const neighborColor = new THREE.Color(0x6366f1);

    // Create word objects
    const wordObjects = [];

    points.forEach(point => {
        const isSeed = seedWords.includes(point.word);
        const size = isSeed ? 0.6 : 0.3;

        // Sphere
        const geometry = new THREE.SphereGeometry(size, 16, 16);
        const material = new THREE.MeshBasicMaterial({
            color: isSeed ? seedColor : neighborColor,
            transparent: true,
            opacity: isSeed ? 1 : 0.7
        });
        const sphere = new THREE.Mesh(geometry, material);

        sphere.position.x = normalize(point.x, xExtent);
        sphere.position.y = normalize(point.y, yExtent);
        sphere.position.z = normalize(point.z, zExtent);

        sphere.userData = { word: point.word, isSeed };

        cloudScene.add(sphere);
        wordObjects.push(sphere);

        // Add glow for seed words
        if (isSeed) {
            const glowGeometry = new THREE.SphereGeometry(size * 1.5, 16, 16);
            const glowMaterial = new THREE.MeshBasicMaterial({
                color: seedColor,
                transparent: true,
                opacity: 0.2
            });
            const glow = new THREE.Mesh(glowGeometry, glowMaterial);
            glow.position.copy(sphere.position);
            cloudScene.add(glow);
        }
    });

    // Create text sprites
    points.forEach(point => {
        const sprite = createTextSprite(point.word, seedWords.includes(point.word));
        sprite.position.x = normalize(point.x, xExtent);
        sprite.position.y = normalize(point.y, yExtent) + 0.8;
        sprite.position.z = normalize(point.z, zExtent);
        cloudScene.add(sprite);
    });

    // Add connecting lines for seed words
    const seedPoints = points.filter(p => seedWords.includes(p.word));
    for (let i = 0; i < seedPoints.length; i++) {
        for (let j = i + 1; j < seedPoints.length; j++) {
            const p1 = seedPoints[i];
            const p2 = seedPoints[j];

            const geometry = new THREE.BufferGeometry().setFromPoints([
                new THREE.Vector3(
                    normalize(p1.x, xExtent),
                    normalize(p1.y, yExtent),
                    normalize(p1.z, zExtent)
                ),
                new THREE.Vector3(
                    normalize(p2.x, xExtent),
                    normalize(p2.y, yExtent),
                    normalize(p2.z, zExtent)
                )
            ]);

            const material = new THREE.LineBasicMaterial({
                color: 0x6366f1,
                transparent: true,
                opacity: 0.3
            });

            const line = new THREE.Line(geometry, material);
            cloudScene.add(line);
        }
    }

    // Ambient particles
    const particleGeometry = new THREE.BufferGeometry();
    const particleCount = 200;
    const positions = new Float32Array(particleCount * 3);

    for (let i = 0; i < particleCount * 3; i++) {
        positions[i] = (Math.random() - 0.5) * 40;
    }

    particleGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));

    const particleMaterial = new THREE.PointsMaterial({
        color: 0x6366f1,
        size: 0.1,
        transparent: true,
        opacity: 0.5
    });

    const particles = new THREE.Points(particleGeometry, particleMaterial);
    cloudScene.add(particles);

    // Animation loop
    function animate() {
        cloudAnimationId = requestAnimationFrame(animate);

        // Rotate particles slowly
        particles.rotation.y += 0.0005;

        cloudControls.update();
        cloudRenderer.render(cloudScene, cloudCamera);
    }

    animate();

    // Handle resize
    window.addEventListener('resize', () => {
        const newWidth = container.clientWidth;
        cloudCamera.aspect = newWidth / height;
        cloudCamera.updateProjectionMatrix();
        cloudRenderer.setSize(newWidth, height);
    });
}

function createTextSprite(text, isSeed) {
    const canvas = document.createElement('canvas');
    const context = canvas.getContext('2d');

    canvas.width = 256;
    canvas.height = 64;

    context.font = isSeed ? 'bold 24px Inter' : '18px Inter';
    context.fillStyle = isSeed ? '#ffffff' : '#a0a0b0';
    context.textAlign = 'center';
    context.textBaseline = 'middle';
    context.fillText(text, 128, 32);

    const texture = new THREE.CanvasTexture(canvas);
    texture.minFilter = THREE.LinearFilter;

    const material = new THREE.SpriteMaterial({
        map: texture,
        transparent: true,
        depthWrite: false
    });

    const sprite = new THREE.Sprite(material);
    sprite.scale.set(4, 1, 1);

    return sprite;
}
