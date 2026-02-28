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
});

