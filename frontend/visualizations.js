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
        .force('link', d3.forceLink(links).id(d => d.id)
            .distance(d => 80 * (1 - d.weight + 0.3))  // Mots proches = liens courts
            .strength(0.8))
        .force('charge', d3.forceManyBody().strength(-200).distanceMax(300))
        .force('center', d3.forceCenter(width / 2, height / 2))
        .force('x', d3.forceX(width / 2).strength(0.05))
        .force('y', d3.forceY(height / 2).strength(0.05))
        .force('collision', d3.forceCollide().radius(45));

    // Zoom behavior
    const zoom = d3.zoom()
        .scaleExtent([0.3, 3])
        .on('zoom', (event) => {
            rootG.attr('transform', event.transform);
        });

    svg.call(zoom);

    // Root group for zoom/pan
    const rootG = svg.append('g');

    // Draw links
    const link = rootG.append('g')
        .selectAll('line')
        .data(links)
        .enter()
        .append('line')
        .attr('class', 'graph-link')
        .attr('stroke-width', d => Math.max(1, d.weight * 3))
        .attr('stroke', 'url(#linkGradient)')
        .attr('stroke-opacity', d => 0.2 + d.weight * 0.5);

    // Draw nodes
    const node = rootG.append('g')
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

    // Auto zoom-to-fit quand la simulation se stabilise
    simulation.on('end', () => {
        const padding = 50;
        const bounds = rootG.node().getBBox();
        const fullWidth = width;
        const fullHeight = height;
        const bWidth = bounds.width;
        const bHeight = bounds.height;

        const scale = Math.min(
            (fullWidth - padding * 2) / bWidth,
            (fullHeight - padding * 2) / bHeight,
            1.5  // Ne pas zoomer trop
        );

        const tx = (fullWidth - bWidth * scale) / 2 - bounds.x * scale;
        const ty = (fullHeight - bHeight * scale) / 2 - bounds.y * scale;

        svg.transition().duration(750).call(
            zoom.transform,
            d3.zoomIdentity.translate(tx, ty).scale(scale)
        );
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
    const height = 500;

    // Scene setup
    cloudScene = new THREE.Scene();
    cloudScene.background = new THREE.Color(0x0f0f1a);

    // Camera - un peu plus de recul pour tout voir
    cloudCamera = new THREE.PerspectiveCamera(60, width / height, 0.1, 1000);
    cloudCamera.position.set(0, 5, 35);

    // Renderer
    cloudRenderer = new THREE.WebGLRenderer({ antialias: true });
    cloudRenderer.setSize(width, height);
    cloudRenderer.setPixelRatio(window.devicePixelRatio);
    container.appendChild(cloudRenderer.domElement);

    // Controls
    cloudControls = new THREE.OrbitControls(cloudCamera, cloudRenderer.domElement);
    cloudControls.enableDamping = true;
    cloudControls.dampingFactor = 0.08;
    cloudControls.autoRotate = true;
    cloudControls.autoRotateSpeed = 0.8;

    // Normalize coordinates - plus d'espace entre les points
    const xExtent = d3.extent(points, d => d.x);
    const yExtent = d3.extent(points, d => d.y);
    const zExtent = d3.extent(points, d => d.z);

    const scale = 25;
    const normalize = (val, extent) => {
        const range = extent[1] - extent[0];
        return range === 0 ? 0 : ((val - extent[0]) / range - 0.5) * scale;
    };

    // Couleurs plus distinctives
    const seedColor = new THREE.Color(0xf59e0b);     // Or pour les seeds
    const neighborColor = new THREE.Color(0x8b5cf6);  // Violet pour les voisins

    // Create word objects
    const wordObjects = [];
    const textSprites = [];

    points.forEach(point => {
        const isSeed = seedWords.includes(point.word);
        const size = isSeed ? 0.9 : 0.45;

        // Sphere
        const geometry = new THREE.SphereGeometry(size, 24, 24);
        const material = new THREE.MeshBasicMaterial({
            color: isSeed ? seedColor : neighborColor,
            transparent: true,
            opacity: isSeed ? 1 : 0.6
        });
        const sphere = new THREE.Mesh(geometry, material);

        sphere.position.x = normalize(point.x, xExtent);
        sphere.position.y = normalize(point.y, yExtent);
        sphere.position.z = normalize(point.z, zExtent);

        sphere.userData = { word: point.word, isSeed };

        cloudScene.add(sphere);
        wordObjects.push(sphere);

        // Glow pour les seed words
        if (isSeed) {
            const glowGeometry = new THREE.SphereGeometry(size * 2, 24, 24);
            const glowMaterial = new THREE.MeshBasicMaterial({
                color: seedColor,
                transparent: true,
                opacity: 0.15
            });
            const glow = new THREE.Mesh(glowGeometry, glowMaterial);
            glow.position.copy(sphere.position);
            cloudScene.add(glow);
        }
    });

    // Create text sprites - plus hauts au dessus des sphères
    points.forEach(point => {
        const isSeed = seedWords.includes(point.word);
        const sprite = createTextSprite(point.word, isSeed);
        sprite.position.x = normalize(point.x, xExtent);
        sprite.position.y = normalize(point.y, yExtent) + (isSeed ? 1.5 : 1.0);
        sprite.position.z = normalize(point.z, zExtent);
        cloudScene.add(sprite);
        textSprites.push(sprite);
    });

    // Lignes de connexion seed -> voisins proches
    const seedPts = points.filter(p => seedWords.includes(p.word));
    const neighborPts = points.filter(p => !seedWords.includes(p.word));

    seedPts.forEach(seed => {
        const seedPos = new THREE.Vector3(
            normalize(seed.x, xExtent),
            normalize(seed.y, yExtent),
            normalize(seed.z, zExtent)
        );

        // Trouver les 5 voisins les plus proches de ce seed
        const distances = neighborPts.map(n => {
            const nPos = new THREE.Vector3(
                normalize(n.x, xExtent),
                normalize(n.y, yExtent),
                normalize(n.z, zExtent)
            );
            return { point: n, dist: seedPos.distanceTo(nPos), pos: nPos };
        }).sort((a, b) => a.dist - b.dist).slice(0, 5);

        distances.forEach(({ pos, dist }) => {
            const geometry = new THREE.BufferGeometry().setFromPoints([seedPos, pos]);
            const material = new THREE.LineBasicMaterial({
                color: 0xf59e0b,
                transparent: true,
                opacity: Math.max(0.08, 0.3 - dist * 0.02)
            });
            const line = new THREE.Line(geometry, material);
            cloudScene.add(line);
        });
    });

    // Lignes entre seeds
    for (let i = 0; i < seedPts.length; i++) {
        for (let j = i + 1; j < seedPts.length; j++) {
            const p1 = seedPts[i];
            const p2 = seedPts[j];

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
                color: 0xf59e0b,
                transparent: true,
                opacity: 0.4
            });

            const line = new THREE.Line(geometry, material);
            cloudScene.add(line);
        }
    }

    // Ambient particles - plus subtils
    const particleGeometry = new THREE.BufferGeometry();
    const particleCount = 150;
    const positions = new Float32Array(particleCount * 3);

    for (let i = 0; i < particleCount * 3; i++) {
        positions[i] = (Math.random() - 0.5) * 50;
    }

    particleGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));

    const particleMaterial = new THREE.PointsMaterial({
        color: 0x6366f1,
        size: 0.08,
        transparent: true,
        opacity: 0.3
    });

    const particles = new THREE.Points(particleGeometry, particleMaterial);
    cloudScene.add(particles);

    // Animation loop
    function animate() {
        cloudAnimationId = requestAnimationFrame(animate);

        // Rotation lente des particules
        particles.rotation.y += 0.0003;

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

    // Résolution doublée pour du texte plus net
    canvas.width = 512;
    canvas.height = 128;

    // Fond semi-transparent pour contraste
    context.fillStyle = 'rgba(15, 15, 26, 0.6)';
    const metrics = context.measureText(text);
    context.font = isSeed ? 'bold 36px Inter, Arial' : '28px Inter, Arial';
    const textWidth = context.measureText(text).width;
    const padding = 16;
    context.beginPath();
    context.roundRect(
        (512 - textWidth) / 2 - padding,
        isSeed ? 30 : 36,
        textWidth + padding * 2,
        isSeed ? 50 : 42,
        8
    );
    context.fill();

    // Texte
    context.fillStyle = isSeed ? '#fbbf24' : '#c4b5fd';
    context.textAlign = 'center';
    context.textBaseline = 'middle';
    context.fillText(text, 256, 58);

    const texture = new THREE.CanvasTexture(canvas);
    texture.minFilter = THREE.LinearFilter;

    const material = new THREE.SpriteMaterial({
        map: texture,
        transparent: true,
        depthWrite: false
    });

    const sprite = new THREE.Sprite(material);
    sprite.scale.set(isSeed ? 7 : 5, isSeed ? 1.8 : 1.3, 1);

    return sprite;
}
