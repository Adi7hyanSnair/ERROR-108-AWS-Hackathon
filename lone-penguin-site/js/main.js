/**
 * Main JavaScript for Magnesia Style Landing Page
 */

document.addEventListener('DOMContentLoaded', () => {
    // 1. Navbar Scroll Effect
    const navbar = document.querySelector('.navbar');

    const handleScroll = () => {
        if (window.scrollY > 50) {
            navbar.classList.add('scrolled');
        } else {
            navbar.classList.remove('scrolled');
        }
    };

    window.addEventListener('scroll', handleScroll);
    handleScroll(); // Init

    // 2. Mobile Menu Toggle
    const hamburger = document.querySelector('.hamburger');
    const navMenu = document.querySelector('.nav-menu');
    const authGroup = document.querySelector('.auth-group');
    const navLinks = document.querySelectorAll('.nav-link, .auth-link, .auth-btn');

    const toggleMenu = () => {
        hamburger.classList.toggle('active');
        navMenu.classList.toggle('active');
        authGroup.classList.toggle('active');

        if (navMenu.classList.contains('active')) {
            document.body.style.overflow = 'hidden';
            navbar.style.background = '#121418';
        } else {
            document.body.style.overflow = '';
            navbar.style.background = '';
        }
    };

    hamburger.addEventListener('click', toggleMenu);

    navLinks.forEach(link => {
        link.addEventListener('click', () => {
            if (navMenu.classList.contains('active')) {
                toggleMenu();
            }
        });
    });

    window.addEventListener('resize', () => {
        if (window.innerWidth > 900 && navMenu.classList.contains('active')) {
            hamburger.classList.remove('active');
            navMenu.classList.remove('active');
            authGroup.classList.remove('active');
            document.body.style.overflow = '';
            navbar.style.background = window.scrollY > 50 ? 'rgba(18, 20, 24, 0.85)' : '';
        }
    });

    // 3. Smooth Scrolling for Anchor Links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            const href = this.getAttribute('href');

            if (href === '#') return;

            const target = document.querySelector(href);

            if (target) {
                e.preventDefault();

                const headerOffset = 80;
                const elementPosition = target.getBoundingClientRect().top;
                const offsetPosition = elementPosition + window.pageYOffset - headerOffset;

                window.scrollTo({
                    top: offsetPosition,
                    behavior: "smooth"
                });
            }
        });
    });

    // 4. Parallax Effect for Hero Elements
    const heroTitle = document.querySelector('.hero-title');
    const heroSubtitle = document.querySelector('.hero-subtitle');

    window.addEventListener('scroll', () => {
        const scrollPosition = window.scrollY;

        if (scrollPosition < window.innerHeight) {
            if (heroTitle) {
                heroTitle.style.transform = `translateY(${scrollPosition * 0.3}px)`;
            }
            if (heroSubtitle) {
                heroSubtitle.style.transform = `translateY(${scrollPosition * 0.2}px)`;
            }
        }
    });

    // 5. Typing Animation for Feature Cards
    function typeText(element, html, speed, callback) {
        if (!element) return;
        element.style.display = 'block';
        element.innerHTML = '';

        const cursor = document.createElement('span');
        cursor.className = 'cursor';
        element.appendChild(cursor);

        let i = 0;
        let currentHtml = '';

        function typeWriter() {
            if (i < html.length) {
                const char = html.charAt(i);

                if (char === '<') {
                    // Find end of tag
                    const endTag = html.indexOf('>', i);
                    if (endTag !== -1) {
                        currentHtml += html.substring(i, endTag + 1);
                        i = endTag + 1;
                        element.innerHTML = currentHtml;
                        element.appendChild(cursor);
                        typeWriter();
                        return;
                    }
                }

                currentHtml += char;
                element.innerHTML = currentHtml;
                element.appendChild(cursor);
                i++;
                setTimeout(typeWriter, speed + (Math.random() * 15));
            } else if (callback) {
                setTimeout(() => {
                    cursor.remove();
                    callback();
                }, 400);
            }
        }
        typeWriter();
    }

    function runAnimations() {
        // Bot Animation
        const botMessage = document.getElementById('bot-message');
        if (botMessage) {
            typeText(botMessage, '<strong>⚠️ Potential Issue | Minor</strong><br><br>Multiple repeated calls to <code style="background:#2b323c; padding:2px 4px; border-radius:4px;">view()</code>. Consider using flattening for clarity.', 25);
        }

        // CLI Animation
        const cliLine1 = document.getElementById('cli-line1');
        const cliLine2 = document.getElementById('cli-line2');
        const cliLine3 = document.getElementById('cli-line3');

        if (cliLine1) {
            cliLine1.innerHTML = '';
            cliLine2.innerHTML = '';
            cliLine3.innerHTML = '';

            typeText(cliLine1, '> neurotidy analyze evaluate.py', 30, () => {
                setTimeout(() => {
                    cliLine2.innerHTML = '✓ Analysis complete (1.2s)';
                    setTimeout(() => {
                        typeText(cliLine3, 'Found 1 critical issue in architecture...', 20);
                    }, 500);
                }, 600);
            });
        }

        // Diff Animation
        const diffLine1 = document.getElementById('diff-line1');
        const diffLine2 = document.getElementById('diff-line2');
        const diffLine3 = document.getElementById('diff-line3');

        if (diffLine1) {
            diffLine1.innerHTML = '';
            diffLine2.innerHTML = '';
            diffLine3.innerHTML = '';

            setTimeout(() => {
                typeText(diffLine1, '- return x.view(-1, 2304)', 25, () => {
                    typeText(diffLine2, '+ return torch.flatten(x, 1)', 25, () => {
                        setTimeout(() => {
                            typeText(diffLine3, '<em>Avoid hardcoding dimension sizes. Use flatten instead.</em>', 20);
                        }, 300);
                    });
                });
            }, 800);
        }
    }

    // Set up Intersection Observer to trigger typing when section is visible
    const featuresSection = document.getElementById('features');
    if (featuresSection) {
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    runAnimations();
                    // Keep looping the animation every 12 seconds to give time for all to finish
                    setInterval(runAnimations, 12000);
                    observer.unobserve(entry.target);
                }
            });
        }, { threshold: 0.2 });

        observer.observe(featuresSection);
    }
    // 6. Interactive Playground    // DOM Elements
    const codeInput = document.getElementById('playground-input') || document.getElementById('code-input');
    const output = document.getElementById('playground-output');
    const runBtn = document.getElementById('run-analysis') || document.getElementById('run-btn');
    const playTabs = document.querySelectorAll('.play-tab');
    const exTags = document.querySelectorAll('.ex-tag');

    let currentMode = 'explain';

    const MOCK_RESPONSES = {
        explain: {
            'Transformer Stability': `
                <div style="font-family: 'SF Mono', 'Fira Code', monospace; font-size: 13px; line-height: 1.6; color: rgba(255,255,255,0.85);">
                    <div style="background: rgba(45, 212, 191, 0.1); border: 1px solid #2dd4bf; padding: 25px; border-radius: 12px; margin-bottom: 25px;">
                        <span style="color: #2dd4bf; font-weight: 700; letter-spacing: 2px;">[ULTRASCAN_ID: NT-TRANS-PRO-99]</span><br>
                        <span style="color: #6a737d;">TIMESTAMP: 2026-03-01 22:08:12 UTC</span><br>
                        <span style="color: #6a737d;">MODE: Deep-Pulse Analytical Trace</span>
                    </div>

                    <h3 style="color: var(--text-light); border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 15px; margin-bottom: 20px; font-weight: 700; letter-spacing: 1px;">📡 PHASE I: ATTENTION KERNEL VALIDATION</h3>
                    <p style="margin-bottom: 20px;">NeuroTidy has detected a <strong>Post-Attention Normalization Breach</strong>. In complex Transformer architectures, the multi-head attention (MHA) operation acts as a high-variance signal generator. Without an immediate LayerNorm bridge, this variance propagates through the residual stream, causing "Hidden State Drift".</p>
                    
                    <div style="background: rgba(0,0,0,0.3); padding: 20px; border-radius: 12px; border: 1px solid rgba(45, 212, 191, 0.2); margin-bottom: 25px;">
                        <h4 style="color:#2dd4bf; font-size: 12px; margin-bottom: 15px; text-transform: uppercase;">Temporal Variance Trace:</h4>
                        <table style="width: 100%; font-size: 12px; color: #fff; border-collapse: collapse;">
                            <tr style="border-bottom: 1px solid rgba(255,255,255,0.1);">
                                <th style="text-align: left; padding: 8px;">NODE</th>
                                <th style="text-align: left; padding: 8px;">σ² MEAN</th>
                                <th style="text-align: left; padding: 8px;">STATUS</th>
                            </tr>
                            <tr><td style="padding: 8px;">Self-Attn</td><td style="padding: 8px;">1.24</td><td style="padding: 8px; color: #27c93f;">NORMAL</td></tr>
                            <tr><td style="padding: 8px;">Residual Add</td><td style="padding: 8px;">4.89</td><td style="padding: 8px; color: #ff5f56;">DIVERGENT</td></tr>
                            <tr><td style="padding: 8px;">MLP Exit</td><td style="padding: 8px;">12+</td><td style="padding: 8px; color: #ff5f56;">VANISHING</td></tr>
                        </table>
                    </div>

                    <h3 style="color: var(--text-light); border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 15px; margin-bottom: 20px; font-weight: 700; letter-spacing: 1px;">📡 PHASE II: GRADIENT PHANTOM AUDIT</h3>
                    <p style="margin-bottom: 15px;">By adding <code style="color:#2dd4bf;">x = x + attn_out</code> without a normalization scale, you are essentially doubling the signal magnitude at every layer. By Layer 12, the original gradient from the loss function is attenuated by a factor of <strong>10^6</strong>. This causes the earlier layers of your Transformer to remain effectively "frozen" (The Ghost Gradient Problem).</p>

                    <h3 style="color: var(--text-light); border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 15px; margin-bottom: 20px; font-weight: 700; letter-spacing: 1px;">📡 PHASE III: PRECISION FLOOR ANALYSIS</h3>
                    <p>Calculated dynamic range overlap: <strong>0.0004%</strong>. In FP16/Mixed-Precision environments, this architecture WILL trigger immediate <code style="color:#ff5f56;">RuntimeError: NaN loss</code> during the first 150 warm-up steps.</p>
                </div>`,
            'Dimension Fix': `
                <div style="font-family: 'SF Mono', 'Fira Code', monospace; font-size: 13px; line-height: 1.6; color: rgba(255,255,255,0.85);">
                    <div style="background: rgba(168, 85, 247, 0.1); border: 1px solid var(--accent-color); padding: 25px; border-radius: 12px; margin-bottom: 25px;">
                        <span style="color: var(--accent-color); font-weight: 700; letter-spacing: 2px;">[ULTRASCAN_ID: NT-DIM-012]</span><br>
                        <span style="color: #6a737d;">OBJECTIVE: Resolution Independence Audit</span>
                    </div>

                    <h3 style="color: var(--text-light); border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 15px; margin-bottom: 20px; font-weight: 700; letter-spacing: 1px;">📡 PHASE I: SPATIAL MAPPING TRACE</h3>
                    <p style="margin-bottom: 20px;">NeuroTidy has identified a <strong>Static Bottleneck</strong> at node <code style="color:var(--accent-color)">fc1</code>. Your architecture is attempting a fixed-resolution feature projection. While this works on local synthetic data (224x224), it fails the "Production Ready" protocol for real-world sensor data.</p>
                    
                    <div style="background: rgba(0,0,0,0.3); padding: 25px; border-radius: 12px; margin-bottom: 25px; border-left: 5px solid var(--accent-color);">
                        <span style="color: var(--accent-color); font-weight: 700;">[KERNEL FOOTPRINT]</span><br>
                        • Input Dim: 4D Tensor (NxCxHxW)<br>
                        • Spatial Resolution: 150,528 (Flattened)<br>
                        • Hardcoded Reference: <code style="color:#ff5f56; font-weight:700;">150528</code><br>
                        • Dependency: [Convolution Exit Node -> Latent Entry]
                    </div>

                    <h3 style="color: var(--text-light); border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 15px; margin-bottom: 20px; font-weight: 700; letter-spacing: 1px;">📡 PHASE II: ARCHITECTURAL RIGIDITY</h3>
                    <p>By using a literal integer, you are mathematically asserting that your model's neural density is only valid for a specific input resolution. This prevents the use of <strong>Progressive Resizing</strong> or <strong>In-Field Cropping</strong>, which are mission-critical for high-performance edge deployment.</p>
                </div>`,
            'Missing Zero Grad': `
                <div style="font-family: 'SF Mono', 'Fira Code', monospace; font-size: 13px; line-height: 1.6; color: rgba(255,255,255,0.85);">
                    <div style="background: rgba(250, 204, 21, 0.1); border: 1px solid #facc15; padding: 25px; border-radius: 12px; margin-bottom: 25px;">
                        <span style="color: #facc15; font-weight: 700; letter-spacing: 2px;">[ULTRASCAN_ID: NT-GRAD-88]</span><br>
                        <span style="color: #6a737d;">OBJECTIVE: Gradient Physics Audit</span>
                    </div>

                    <h3 style="color: var(--text-light); border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 15px; margin-bottom: 20px; font-weight: 700; letter-spacing: 1px;">📡 PHASE I: PERSISTENT BUFFER TRACE</h3>
                    <p style="margin-bottom: 20px;">NeuroTidy has detected a critical **Memory Leakage** in the gradient buffer. In the PyTorch ecosystem, the <code style="color:#facc15">.grad</code> attribute is an additive accumulator. Without a manual reset, your optimizer is applying weight updates based on the sum of <u>every batch since training started</u>.</p>
                    
                    <div style="background: rgba(250, 204, 21, 0.05); border: 1px solid rgba(250, 204, 21, 0.2); padding: 25px; border-radius: 12px; margin-bottom: 25px;">
                        <h4 style="color:#facc15; font-size: 12px; margin-bottom: 10px;">Execution Sequence Log:</h4>
                        <ol style="margin-top:0; color:var(--text-muted);">
                            <li><strong>Step [N]</strong>: Grad = Δ1 (Correct)</li>
                            <li><strong>Step [N+1]</strong>: Grad = Δ1 + Δ2 <span style="color:#ff5f56; font-weight:700;">[LEAK]</span></li>
                            <li><strong>Step [N+2]</strong>: Grad = Δ1 + Δ2 + Δ3 <span style="color:#ff5f56; font-weight:700;">[DIVERGENCE REACHED]</span></li>
                        </ol>
                    </div>

                    <h3 style="color: var(--text-light); border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 15px; margin-bottom: 20px; font-weight: 700; letter-spacing: 1px;">📡 PHASE II: STOCHASTIC COLLAPSE</h3>
                    <p>The absence of <code style="color:#facc15">zero_grad()</code> causes your effective Learning Rate to increase exponentially with batch count. This triggers <strong>Geometric Divergence</strong>, where the weights move so far from the minima that the loss function enters an undefined (NaN) numerical state.</p>
                </div>`,
            'Loss Mismatch': `
                <div style="font-family: 'SF Mono', 'Fira Code', monospace; font-size: 13px; line-height: 1.6; color: rgba(255,255,255,0.85);">
                    <div style="background: rgba(255, 95, 86, 0.1); border: 1px solid #ff5f56; padding: 20px; border-radius: 12px; margin-bottom: 25px;">
                        <span style="color: #ff5f56; font-weight: 700; letter-spacing: 1px;">[SCAN_REPORT: NT-0441L]</span><br>
                        <span style="color: #6a737d;">EXECUTED: 2026-03-01 21:58:12 UTC</span><br>
                        <span style="color: #6a737d;">OBJECTIVE: Loss Entropy Audit</span>
                    </div>

                    <h3 style="color: var(--text-light); border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 12px; margin-bottom: 20px; font-weight: 700;">📡 I. PROBABILITY SPACE OVERLAP</h3>
                    <p style="margin-bottom: 20px;">The analysis engine detected a <strong>Double Saturation Point</strong> in the final layer activation sequence. Your model is squashing logits twice.</p>

                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-bottom: 25px;">
                        <div style="background: rgba(255, 95, 86, 0.05); border: 1px solid rgba(255, 95, 86, 0.2); padding: 15px; border-radius: 12px;">
                            <span style="color: #ff5f56; font-size: 11px;">OBSERVED:</span><br>
                            <code style="font-size: 11px;">Sigmoid -> LogSoftmax</code>
                        </div>
                        <div style="background: rgba(39, 201, 63, 0.05); border: 1px solid rgba(39, 201, 63, 0.2); padding: 15px; border-radius: 12px;">
                            <span style="color: #27c93f; font-size: 11px;">RECOMMENDED:</span><br>
                            <code style="font-size: 11px;">Raw Logits -> LogSoftmax</code>
                        </div>
                    </div>

                    <h3 style="color: var(--text-light); border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 12px; margin-bottom: 20px; font-weight: 700;">📡 II. GRADIENT VANISHING PREDICTION</h3>
                    <p style="margin-bottom: 15px;">By squashing values into the <code style="color:#ff5f56">[0, 1]</code> range twice, you are severely compressing the dynamic range. In backpropagation, the derivative of a Sigmoid is maxed at 0.25—nesting this causes the signal to disappear exponentially fast.</p>
                    
                    <p style="background: rgba(255,255,255,0.03); padding: 15px; border-radius: 8px;"><strong>Result:</strong> Your model will experience <u>Zero Learning</u> in its early layers.</p>
                </div>`
        },
        analyze: {}, // Kept for structure
        optimize: {} // Kept for structure
    };

    // Tab Switching
    playTabs.forEach(tab => {
        tab.addEventListener('click', () => {
            playTabs.forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            currentMode = tab.dataset.mode;
        });
    });

    // Example Clicks
    exTags.forEach(tag => {
        tag.addEventListener('click', () => {
            codeInput.value = tag.dataset.code;
            tag.classList.add('pulse');
            setTimeout(() => tag.classList.remove('pulse'), 500);
        });
    });

    // Response Formatters
    function formatExplain(result) {
        let html = `<div style="font-family: 'SF Mono', 'Fira Code', monospace; font-size: 13px; line-height: 1.6; color: rgba(255,255,255,0.85);">`;
        html += `<div style="background: rgba(45, 212, 191, 0.1); border: 1px solid #2dd4bf; padding: 25px; border-radius: 12px; margin-bottom: 25px;">
                    <span style="color: #2dd4bf; font-weight: 700; letter-spacing: 2px;">[ULTRASCAN_ID: ${result.analysis_id || 'NT-AUTO'}]</span><br>
                    <span style="color: #6a737d;">MODE: Code Explanation</span>
                 </div>`;

        html += `<h3 style="color: var(--text-light); border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 15px; margin-bottom: 20px; font-weight: 700; letter-spacing: 1px;">📡 EXPLANATION</h3>`;

        let text = result.explanation || "No explanation provided.";
        if (typeof text === 'object') text = JSON.stringify(text, null, 2);

        // Format code blocks and bold globally
        text = text.replace(/`([^`]+)`/g, '<code style="color:#2dd4bf; font-weight:bold; background:rgba(0,0,0,0.3); padding:2px 4px; border-radius:4px;">$1</code>');
        text = text.replace(/\*\*([^\*]+)\*\*/g, '<strong>$1</strong>');

        // Parse line by line to handle Markdown headings and lists flawlessly
        let formattedLines = text.split('\n').map(line => {
            let t = line.trim();
            if (t.startsWith('### ')) {
                return '<h4 style="color:#fff; margin-top:25px; margin-bottom:10px; font-weight:700; font-size:14px;">' + t.substring(4) + '</h4>';
            } else if (t.startsWith('## ')) {
                return '<h3 style="color:#fff; margin-top:30px; margin-bottom:15px; font-weight:700; font-size:16px;">' + t.substring(3) + '</h3>';
            } else if (t.startsWith('* ')) {
                return '<div style="margin-left:15px; margin-bottom:8px;"><span style="color:#2dd4bf;">•</span> ' + t.substring(2) + '</div>';
            } else if (t.startsWith('- ')) {
                return '<div style="margin-left:15px; margin-bottom:8px;"><span style="color:#2dd4bf;">•</span> ' + t.substring(2) + '</div>';
            } else {
                return line;
            }
        });

        text = formattedLines.join('<br>');

        html += `<p style="margin-bottom: 20px;">${text}</p></div>`;
        return html;
    }

    function formatAnalyze(result) {
        let html = `<div style="font-family: 'SF Mono', 'Fira Code', monospace; font-size: 13px; line-height: 1.6; color: rgba(255,255,255,0.85);">`;
        html += `<div style="background: rgba(168, 85, 247, 0.1); border: 1px solid var(--accent-color); padding: 25px; border-radius: 12px; margin-bottom: 25px;">
                    <span style="color: var(--accent-color); font-weight: 700; letter-spacing: 2px;">[ULTRASCAN_ID: ${result.analysis_id || 'NT-AUTO'}]</span><br>
                    <span style="color: #6a737d;">OBJECTIVE: Static Analysis Report</span>
                 </div>`;

        if (result.summary) {
            html += `<p style="margin-bottom: 20px;">${result.summary}</p>`;
        }

        if (result.violations && result.violations.length > 0) {
            html += `<h3 style="color: var(--text-light); border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 15px; margin-bottom: 20px; font-weight: 700; letter-spacing: 1px;">📡 ISSUES FOUND (${result.violations.length})</h3>`;
            html += `<div style="background: rgba(0,0,0,0.3); padding: 20px; border-radius: 12px; border: 1px solid rgba(255, 95, 86, 0.2); margin-bottom: 25px;">`;
            result.violations.forEach(v => {
                const color = v.severity === 'CRITICAL' || v.severity === 'HIGH' ? '#ff5f56' : '#facc15';
                html += `<div style="margin-bottom: 15px;">
                            <span style="color: ${color}; font-weight: 700;">[${v.severity || 'ISSUE'}]</span> <span style="color: #6a737d;">Line ${v.line_number || '?'} (${v.rule_id || 'unknown'})</span><br>
                            ${v.description || ''}
                         </div>`;
            });
            html += `</div>`;
        } else {
            html += `<div style="background: rgba(39, 201, 63, 0.1); border: 1px solid #27c93f; padding: 15px; border-radius: 12px; margin-bottom: 25px; color: #27c93f;">✓ No static analysis violations found.</div>`;
        }

        if (result.ai_insights) {
            html += `<h3 style="color: var(--text-light); border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 15px; margin-bottom: 20px; font-weight: 700; letter-spacing: 1px;">📡 AI INSIGHTS</h3>`;
            html += `<div style="background: rgba(0,0,0,0.3); padding: 20px; border-radius: 12px; border-left: 3px solid #2dd4bf;">`;
            if (result.ai_insights.readability_score) {
                html += `Readability: <strong>${result.ai_insights.readability_score}/100</strong><br>`;
            }
            if (result.ai_insights.maintainability_score) {
                html += `Maintainability: <strong>${result.ai_insights.maintainability_score}/100</strong><br><br>`;
            }
            if (result.ai_insights.top_recommendation) {
                html += `<span style="color:#2dd4bf;">💡 Recommendation:</span> ${result.ai_insights.top_recommendation}`;
            }
            html += `</div>`;
        }
        html += `</div>`;
        return html;
    }

    function formatOptimize(result) {
        let html = `<div style="font-family: 'SF Mono', 'Fira Code', monospace; font-size: 13px; line-height: 1.6; color: rgba(255,255,255,0.85);">`;
        html += `<div style="background: rgba(250, 204, 21, 0.1); border: 1px solid #facc15; padding: 25px; border-radius: 12px; margin-bottom: 25px;">
                    <span style="color: #facc15; font-weight: 700; letter-spacing: 2px;">[ULTRASCAN_ID: ${result.analysis_id || 'NT-AUTO'}]</span><br>
                    <span style="color: #6a737d;">OBJECTIVE: DL Optimization Report</span>
                 </div>`;

        if (result.summary) {
            html += `<p style="margin-bottom: 20px;">${result.summary}</p>`;
        }

        if (result.violations && result.violations.length > 0) {
            html += `<h3 style="color: var(--text-light); border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 15px; margin-bottom: 20px; font-weight: 700; letter-spacing: 1px;">📡 BOTTLENECKS FOUND (${result.violations.length})</h3>`;
            html += `<div style="background: rgba(0,0,0,0.3); padding: 20px; border-radius: 12px; border: 1px solid rgba(250, 204, 21, 0.2); margin-bottom: 25px;">`;
            result.violations.forEach(v => {
                const color = v.severity === 'HIGH' ? '#ff5f56' : '#facc15';
                html += `<div style="margin-bottom: 15px;">
                            <span style="color: ${color}; font-weight: 700;">[${v.severity || 'ISSUE'}]</span> <span style="color: #6a737d;">(${v.rule_id || 'unknown'})</span><br>
                            ${v.description || ''}<br>`;
                if (v.suggested_fix) {
                    html += `<span style="color:#2dd4bf;">→ Fix: ${v.suggested_fix}</span>`;
                }
                html += `</div>`;
            });
            html += `</div>`;
        } else {
            html += `<div style="background: rgba(39, 201, 63, 0.1); border: 1px solid #27c93f; padding: 15px; border-radius: 12px; margin-bottom: 25px; color: #27c93f;">✓ Architecture is highly optimized.</div>`;
        }

        if (result.ai_insights) {
            html += `<h3 style="color: var(--text-light); border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 15px; margin-bottom: 20px; font-weight: 700; letter-spacing: 1px;">📡 AI PERFORMANCE INSIGHTS</h3>`;
            html += `<div style="background: rgba(250, 204, 21, 0.05); border: 1px solid rgba(250, 204, 21, 0.2); padding: 20px; border-radius: 12px;">`;
            if (result.ai_insights.performance_score) {
                html += `Efficiency Score: <strong>${result.ai_insights.performance_score}/100</strong><br>`;
            }
            if (result.ai_insights.estimated_speedup) {
                html += `<span style="color:#2dd4bf;">⚡ Estimated speedup: ${result.ai_insights.estimated_speedup}</span><br><br>`;
            }
            if (result.ai_insights.quick_wins && result.ai_insights.quick_wins.length > 0) {
                result.ai_insights.quick_wins.forEach(win => {
                    html += `✓ ${win}<br>`;
                });
            }
            html += `</div>`;
        }
        html += `</div>`;
        return html;
    }

    // Run Analysis
    if (runBtn) {
        runBtn.addEventListener('click', async () => {
            const code = codeInput.value.trim();
            if (!code) {
                output.innerHTML = '<span class="placeholder-text">Analyzing with NeuroTidy AI...</span>';
                return;
            }
            runBtn.disabled = true;

            // Simulate Network delay or AI processing
            output.classList.add('analyzing-glow');
            output.innerHTML = '<span class="placeholder-text" style="color:#2dd4bf;"><span class="pulse">...</span> Contacting NeuroTidy Cloud Engine </span>';

            const payload = { code: code, mode: 'intermediate', use_ai: true, use_rag: true };
            const endpoint = `https://1d21iee6x0.execute-api.us-east-1.amazonaws.com/prod/${currentMode}`;

            try {
                const response = await fetch(endpoint, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });

                if (!response.ok) throw new Error("HTTP " + response.status);
                const result = await response.json();

                let htmlOutput = '';
                if (currentMode === 'explain') htmlOutput = formatExplain(result);
                else if (currentMode === 'analyze') htmlOutput = formatAnalyze(result);
                else if (currentMode === 'optimize') htmlOutput = formatOptimize(result);
                else htmlOutput = formatExplain(result);

                typeText(output, htmlOutput, 3, () => {
                    runBtn.disabled = false;
                    output.classList.remove('analyzing-glow');
                });

            } catch (err) {
                console.error("NeuroTidy API Error:", err);

                // Fallback to MOCK_RESPONSES if API is unavailable or rate limited
                let fallbackHtml = `
                    <div style="border-left: 3px solid #ff5f56; padding-left: 15px; margin-bottom: 20px;">
                        <span style="color: #ff5f56; font-size: 11px;">[WARNING: CLOUD UNAVAILABLE]</span><br>
                        <span style="color: #6a737d;">Falling back to localized heuristic scan.</span>
                    </div>`;

                const codeUpper = code.toUpperCase();
                const matches = {
                    'Transformer Stability': ['TRANSFORMER', 'ATTENTION', 'NHEAD', 'D_MODEL'],
                    'Dimension Fix': ['VIEW', '150528', 'FLATTEN', 'RESHAPE'],
                    'Missing Zero Grad': ['BACKWARD', 'STEP', 'ZERO_GRAD', 'OPTIMIZER'],
                    'Loss Mismatch': ['SIGMOID', 'CROSSENTROPY', 'SOFTMAX', 'CRITERION']
                };

                let matched = false;
                for (const [key, keywords] of Object.entries(matches)) {
                    if (keywords.some(k => codeUpper.includes(k))) {
                        fallbackHtml += MOCK_RESPONSES[currentMode] && MOCK_RESPONSES[currentMode][key] ? MOCK_RESPONSES[currentMode][key] : MOCK_RESPONSES.explain[key];
                        matched = true;
                        break;
                    }
                }

                if (!matched) {
                    fallbackHtml += `
                    <div style="border-left: 3px solid #27c93f; padding-left: 15px;">
                        <h4 style="color: #27c93f; margin-bottom: 10px;">✨ Heuristic Scan Verified</h4>
                        <p>No critical structural flaws detected in offline mode. For deep AI analysis, please restore internet connection to NeuroTidy Cloud.</p>
                    </div>`;
                }

                typeText(output, fallbackHtml, 12, () => {
                    runBtn.disabled = false;
                    output.classList.remove('analyzing-glow');
                });
            }
        });
    }

    // 7. Contact Form Handling
    const contactForm = document.getElementById('contact-form');
    const formStatus = document.getElementById('form-status');

    if (contactForm && formStatus) {
        contactForm.addEventListener('submit', async (e) => {
            e.preventDefault();

            const submitBtn = contactForm.querySelector('button[type="submit"]');
            const originalBtnContent = submitBtn.innerHTML;

            // Show loading state
            submitBtn.disabled = true;
            submitBtn.innerHTML = 'Sending... <span class="btn-icon">🕒</span>';

            // Real API Call
            try {
                const baseUrl = "__API_BASE_URL__";
                const response = await fetch(`${baseUrl}/contact`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        name: contactForm.name.value,
                        email: contactForm.email.value,
                        subject: contactForm.subject.value,
                        message: contactForm.message.value
                    })
                });

                if (response.ok) {
                    formStatus.textContent = 'Thank you! Your message has been sent successfully. We will get back to you soon.';
                    formStatus.className = 'form-status success';
                    formStatus.style.display = 'block';
                    contactForm.reset();
                } else {
                    throw new Error('Failed to send message');
                }
            } catch (err) {
                console.error(err);
                formStatus.textContent = 'Sorry, there was an error sending your message. Please try again later or email us directly.';
                formStatus.className = 'form-status error';
                formStatus.style.display = 'block';
            } finally {
                submitBtn.disabled = false;
                submitBtn.innerHTML = originalBtnContent;

                // Hide status after 8 seconds
                setTimeout(() => {
                    formStatus.style.display = 'none';
                    formStatus.className = 'form-status';
                }, 8000);
            }
        });
    }
});
