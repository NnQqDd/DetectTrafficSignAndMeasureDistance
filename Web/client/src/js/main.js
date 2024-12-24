let lastScrollTop = 0;
const header = document.querySelector('.site-header');
let isScrolling;
const hideDelay = 1000;
let isHovering = false;

header.addEventListener('mouseenter', () => {
  isHovering = true;
  clearTimeout(isScrolling);
  header.classList.remove('hidden');
});

header.addEventListener('mouseleave', () => {
  isHovering = false;
});

window.addEventListener('scroll', () => {
  const scrollTop = window.scrollY;

  // Nếu đang ở đầu trang (scrollTop bằng 0), không cần ẩn menu
  if (scrollTop === 0) {
    header.classList.remove('hidden');
    return; // Không làm gì khác
  }

  if (scrollTop > lastScrollTop) {
    clearTimeout(isScrolling);
    if (!isHovering) {
      isScrolling = setTimeout(() => {
        header.classList.add('hidden');
      }, hideDelay);
    }
  } else {
    header.classList.remove('hidden');
  }

  lastScrollTop = scrollTop <= 0 ? 0 : scrollTop;
});




document.querySelector('.navbar-toggler').addEventListener('click', function () {
  var navbarMenu = document.querySelector('.navbar-mobie');
  navbarMenu.classList.toggle('navbar-mobie-active');
});
(function () {
  const win = window
  const doc = document.documentElement

  doc.classList.remove('no-js')
  doc.classList.add('js')

  // Reveal animations
  if (document.body.classList.contains('has-animations')) {
    /* global ScrollReveal */
    const sr = window.sr = ScrollReveal()

    sr.reveal('.feature, .pricing-table-inner', {
      duration: 600,
      distance: '20px',
      easing: 'cubic-bezier(0.5, -0.01, 0, 1.005)',
      origin: 'bottom',
      interval: 100
    })

    doc.classList.add('anime-ready')
    /* global anime */
    anime.timeline({
      targets: '.hero-figure-box-05'
    }).add({
      duration: 400,
      easing: 'easeInOutExpo',
      scaleX: [0.05, 0.05],
      scaleY: [0, 1],
      perspective: '500px',
      delay: anime.random(0, 400)
    }).add({
      duration: 400,
      easing: 'easeInOutExpo',
      scaleX: 1
    }).add({
      duration: 800,
      rotateY: '-15deg',
      rotateX: '8deg',
      rotateZ: '-1deg'
    })

    anime.timeline({
      targets: '.hero-figure-box-06, .hero-figure-box-07'
    }).add({
      duration: 400,
      easing: 'easeInOutExpo',
      scaleX: [0.05, 0.05],
      scaleY: [0, 1],
      perspective: '500px',
      delay: anime.random(0, 400)
    }).add({
      duration: 400,
      easing: 'easeInOutExpo',
      scaleX: 1
    }).add({
      duration: 800,
      rotateZ: '20deg'
    })

    anime({
      targets: '.hero-figure-box-01, .hero-figure-box-02, .hero-figure-box-03, .hero-figure-box-04, .hero-figure-box-08, .hero-figure-box-09, .hero-figure-box-10',
      duration: anime.random(600, 800),
      delay: anime.random(600, 800),
      rotate: [anime.random(-360, 360), function (el) { return el.getAttribute('data-rotation') }],
      scale: [0.7, 1],
      opacity: [0, 1],
      easing: 'easeInOutExpo'
    })
  }
}())


document.addEventListener('DOMContentLoaded', function () {
  var parent = document.querySelector('.splitview'),
    topPanel = parent.querySelector('.top'),
    handle = parent.querySelector('.handle'),
    skewHack = 0,
    delta = 0;

  // If the parent has .skewed class, set the skewHack var.
  if (parent.className.indexOf('skewed') != -1) {
    skewHack = 1000;
  }

  parent.addEventListener('mousemove', function (event) {
    // Get the delta between the mouse position and center point.
    delta = (event.clientX - window.innerWidth / 2) * 0.5;

    // Move the handle.
    handle.style.left = event.clientX + delta + 'px';

    // Adjust the top panel width.
    topPanel.style.width = event.clientX + skewHack + delta + 'px';
  });
});


document.getElementById('input-file-img').addEventListener('change', function (event) {
  const file = event.target.files[0];
  const maxSize = 5 * 1024 * 1024; // 5MB
  const resultImage = document.querySelector('.result-image');
  resultImage.style.backgroundImage = 'url(../../../dist/images/result-placeholder.svg)';
  resultImage.style.cursor = 'none';
  resultImage.classList.add('no-zoom');
  if (file && (file.type === 'image/jpeg' || file.type === 'image/jpg')) {
    if (file.size <= maxSize) {
      const reader = new FileReader();

      reader.onload = function (e) {
        const img = document.getElementById('preview-image');
        img.src = e.target.result; // Gán URL base64 cho ảnh
        img.style.display = 'block';
        document.getElementById('tryAI-button').disabled = false;
      };

      reader.readAsDataURL(file); // Đọc file và chuyển sang URL base64
    } else {
      alert('File size must be less than 5MB.');
      event.target.value = ''; // Xóa file khỏi input nếu không hợp lệ
    }
  } else {
    alert('Please upload a valid JPG/JPEG image.');
    event.target.value = ''; // Xóa file nếu không đúng định dạng
  }
});
document.querySelector('.btn-del').addEventListener('click', function () {
  const img = document.getElementById('preview-image');
  img.src = './dist/images/uploadImg.png';
  img.style.display = 'block';
  document.getElementById('input-file-img').value = '';
  document.getElementById('tryAI-button').disabled = true;
  const resultImage = document.querySelector('.result-image');
  resultImage.style.backgroundImage = 'url(../../../dist/images/result-placeholder.svg)';
  resultImage.style.cursor = 'none';
  resultImage.classList.add('no-zoom');

});
document.getElementById('tryAI-button').addEventListener('click', async function () {
  const inputFile = document.getElementById('input-file-img');
  const btn = document.getElementById('tryAI-button');
  const files = inputFile.files;

  if (files.length === 0) {
    alert("Please upload an image before submitting!");
    return;
  }

  btn.classList.add('loading-active');
  btn.disabled = true;

  try {
    const file = files[0];
    const response = await uploadImage(file);
    const boxes = await response.json();
    drawImageAndBoxes(file, boxes);
    btn.classList.remove('loading-active');
    btn.disabled = false;
  } catch (error) {
    console.error("Error uploading image:", error);
    alert("Error uploading image.");
    btn.classList.remove('loading-active');
    btn.disabled = false;
  }
});
async function uploadImage(file, timeoutDuration = 20000) {
  const controller = new AbortController();
  const signal = controller.signal;

  const data = new FormData();
  data.append("image_file", file);

  const timeoutId = setTimeout(() => {
    controller.abort();
  }, timeoutDuration);

  try {
    const response = await fetch("http://localhost:8080/detect", {
      method: "POST",
      body: data,
      signal: signal
    });
    console.log(response)
    clearTimeout(timeoutId);

    if (!response.ok) {
      throw new Error('Network response was not ok');
    }

    return response;
  } catch (error) {
    if (error.name === 'AbortError') {
      console.error('Request was aborted');
    } else {
      console.error('Fetch error:', error);
    }
    throw error;
  }
}
function drawImageAndBoxes(file, boxes) {
  const img = new Image();
  img.src = URL.createObjectURL(file);

  img.onload = () => {
    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d");
    canvas.width = img.width;
    canvas.height = img.height;

    ctx.drawImage(img, 0, 0);

    if (boxes.length > 0) {
      ctx.strokeStyle = "#00FF00";
      ctx.lineWidth = 3;
      ctx.font = "18px serif";
      boxes.forEach(([x1, y1, x2, y2, label]) => {
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
        const width = ctx.measureText(label).width;
        ctx.fillStyle = "#00ff00";
        ctx.fillRect(x1, y1, width + 10, 25);
        ctx.fillStyle = "#000000";
        ctx.fillText(label, x1, y1 + 18);
      });
      updateResults(canvas.toDataURL("image/png"), boxes[0][4]);
    } else {
      updateResults(img.src);
    }
  };
}

function updateResults(imageSrc, nameSign = "") {
  const resultImage = document.querySelector('.result-image');
  resultImage.style.backgroundImage = `url(${imageSrc})`;
  resultImage.style.cursor = "pointer";
  resultImage.classList.remove('no-zoom');
}


document.querySelector('.result-image').addEventListener('click', function () {
  const bgImage = getComputedStyle(this).backgroundImage;
  const imageUrl = bgImage.slice(5, -2);
  if (!imageUrl.endsWith('.svg')) {
    document.querySelector('.lightbox img').src = imageUrl;
    document.querySelector('.lightbox').style.display = 'flex';
  }
});

document.querySelector('.lightbox').addEventListener('click', function () {
  this.style.display = 'none';
  this.querySelector('img').src = '';
});

document.querySelector('.result-image').addEventListener('click', function () {
  const bgImage = getComputedStyle(this).backgroundImage;
  const imageUrl = bgImage.slice(5, -2);

  const lightboxImage = document.querySelector('.result-image-zoom');
  lightboxImage.src = imageUrl;
  document.querySelector('.lightbox').style.display = 'flex';
});

document.querySelector('.lightbox').addEventListener('click', function () {
  this.style.display = 'none';
});

const trafic_signs = [
  {
    name: "Vạch qua đường cho người đi bộ",
    url: "./dist/images/trafic_signs/PedestrianCrossing.jpg",
    desc: ""
  },
  {
    name: "Chỉ được rẽ phải",
    url: "./dist/images/trafic_signs/RightTurnOnly.jpg",
    desc: ""
  },
  {
    name: "Đường giao nhau",
    url: "./dist/images/trafic_signs/Intersection.jpg",
    desc: ""
  },
  {
    name: "Cấm rẽ trái",
    url: "./dist/images/trafic_signs/NoLeftTurn.png",
    desc: ""
  },
  {
    name: "Điểm dừng xe bus",
    url: "./dist/images/trafic_signs/BusStop.png",
    desc: ""
  },
  {
    name: "Vòng xuyến",
    url: "./dist/images/trafic_signs/Roundabout.jpg",
    desc: ""
  },
  {
    name: "Cảnh báo có vòng xuyến",
    url: "./dist/images/trafic_signs/Roundabout2.png",
    desc: ""
  },
  {
    name: "Cấm dừng và đậu xe",
    url: "./dist/images/trafic_signs/NoStoppingAndNoParking.png",
    desc: ""
  },
  {
    name: "Cấm quay đầu",
    url: "./dist/images/trafic_signs/UTurnAllowed.png",
    desc: ""
  },
  {
    name: "Phân làn",
    url: "./dist/images/trafic_signs/LaneAllocation.png",
    desc: ""
  },
  {
    name: "Đi chậm",
    url: "./dist/images/trafic_signs/SlowDown.png",
    desc: ""
  },
  {
    name: "Cấm xe tải",
    url: "./dist/images/trafic_signs/NoTrucksAllowed.png",
    desc: ""
  },
  {
    name: "Đường hẹp bên phải",
    url: "./dist/images/trafic_signs/NarrowRoadOnTheRight.png",
    desc: ""
  },
  {
    name: "",
    url: "./dist/images/trafic_signs/NoPassengerCarsAndTrucks.png",
    desc: ""
  },
  {
    name: "Giới hạn chiều cao",
    url: "./dist/images/trafic_signs/HeightLimit.png",
    desc: ""
  },
  {
    name: "Cấm quay đầu trái",
    url: "./dist/images/trafic_signs/NoLeftUTurn.png",
    desc: ""
  },
  {
    name: "Cấm quay đầu phải",
    url: "./dist/images/trafic_signs/NoRightUTurn.png",
    desc: ""
  },
  {
    name: "Cấm quay đầu và cấm rẽ phải",
    url: "./dist/images/trafic_signs/NoUTurnandNoRightTurn.png",
    desc: ""
  },
  {
    name: "Cấm ô tô",
    url: "./dist/images/trafic_signs/NoCarsAllowed.png",
    desc: ""
  },
  {
    name: "Đường hẹp bên trái",
    url: "./dist/images/trafic_signs/NarrowRoadOnTheLeft.png",
    desc: ""
  },
  {
    name: "Đường gồ ghề",
    url: "./dist/images/trafic_signs/UnevenRoad.png",
    desc: ""
  },
  {
    name: "Điểm kiểm tra phương tiện",
    url: "./dist/images/trafic_signs/CustomsCheckpoint.png",
    desc: ""
  },
  {
    name: 'Chỉ xe máy',
    url: './dist/images/trafic_signs/mortocycle-only.png',
    desc: '',
  },
  {
    name: 'Chướng ngại vật trên đường',
    url: './dist/images/trafic_signs/obstancle-on-the-road.png',
    desc: '',
  },
  {
    name: 'Trẻ em có mặt',
    url: './dist/images/trafic_signs/children-present.png',
    desc: '',
  },
  {
    name: 'Không có xe tải và container',
    url: './dist/images/trafic_signs/no-trucks-and-container.png',
    desc: '',
  },
  {
    name: 'Cấm xe máy',
    url: './dist/images/trafic_signs/no-mortocycles.png',
    desc: '',
  },
  {
    name: 'Đường Có Camera Giám Sát',
    url: './dist/images/trafic_signs/road-with-surveillance-camera.png',
    desc: '',
  },
  {
    name: 'Cấm Rẽ Phải',
    url: './dist/images/trafic_signs/no-right-turn.png',
    desc: '',
  },
  {
    name: 'Hàng loạt những ngã rẽ nguy hiểm',
    url: './dist/images/trafic_signs/series-of-dangerous-turns.png',
    desc: '',
  },
  {
    name: 'Không được phép cho xe tải',
    url: './dist/images/trafic_signs/no-container-allowed.png',
    desc: '',
  },
  {
    name: 'Cấm rẽ trái hoặc rẽ phải',
    url: './dist/images/trafic_signs/no-left-or-right-turn.png',
    desc: '',
  },
  {
    name: 'Cấm đi thẳng và rẽ phải',
    url: './dist/images/trafic_signs/no-straight-or-right-turn.png',
    desc: '',
  },
  {
    name: 'Nút giao với ngã ba chữ T',
    url: './dist/images/trafic_signs/T-junction.png',
    desc: '',
  },
  {
    name: 'Giới hạn tốc độ (50km/h)',
    url: './dist/images/trafic_signs/speed-limit-50km.png',
    desc: '',
  },
  {
    name: 'Giới hạn tốc độ (60km/h)',
    url: './dist/images/trafic_signs/speed-limit-60km.png',
    desc: '',
  },
  {
    name: 'Giới hạn tốc độ (80km/h)',
    url: './dist/images/trafic_signs/speed-limit-80km.png',
    desc: '',
  },
  {
    name: 'Giới hạn tốc độ (40km/h)',
    url: './dist/images/trafic_signs/speed-limit-40km.png',
    desc: '',
  },
  {
    name: 'Rẽ trái',
    url: './dist/images/trafic_signs/left-turn.png',
    desc: '',
  },
  {
    name: 'Nguy hiểm khác',
    url: './dist/images/trafic_signs/other-danger.png',
    desc: '',
  },
  {
    name: 'Đi thẳng',
    url: './dist/images/trafic_signs/go-straight.png',
    desc: '',
  },
  {
    name: 'Không đậu xe ',
    url: './dist/images/trafic_signs/no-parking-2.png',
    desc: '',
  },
  {
    name: 'Chỉ vùng chứa Ôtô cấm quay đầu bên phải',
    url: './dist/images/trafic_signs/no-right-turn-U-cars.png',
    desc: '',
  },
  {
    name: 'Vượt cấp với rào cản đường tàu',
    url: './dist/images/trafic_signs/level-crossing-with-barriers.png',
    desc: '',
  },
  {
    name: 'Đèn giao thông (đỏ, xanh, vàng)',
    url: './dist/images/trafic_signs/traffic-light.png',
    desc: '',
  }
]
const trafficSignsList = document.querySelector('#trafficSignsList');
const searchInput = document.querySelector('#searchInput');
const loadMoreButton = document.querySelector('#loadMoreButton');
const initialCount = 6;
let displayedCount = initialCount;

function displayTrafficSigns(signs) {
  trafficSignsList.innerHTML = '';
  const limitedSigns = signs.slice(0, displayedCount);

  limitedSigns.forEach(sign => {
    const signItem = document.createElement('div');
    signItem.className = 'sign-item';
    signItem.innerHTML = `
            <img src="${sign.url}" alt="${sign.name}" />
            <div class="sign-info">
                <h3>${sign.name}</h3>
            </div>
        `;
    trafficSignsList.appendChild(signItem);
  });
  loadMoreButton.style.display = (displayedCount < signs.length) ? 'block' : 'none';
}

function searchTrafficSigns(query) {
  const lowerCaseQuery = query.toLowerCase();
  const filteredSigns = trafic_signs.filter(sign =>
    sign.name.toLowerCase().includes(lowerCaseQuery)
  );
  return filteredSigns;
}

let timeoutId;

searchInput.addEventListener('input', () => {
  clearTimeout(timeoutId);
  timeoutId = setTimeout(() => {
    displayedCount = initialCount;
    const query = searchInput.value;
    const filteredSigns = searchTrafficSigns(query);
    displayTrafficSigns(filteredSigns);
  }, 300);
});

loadMoreButton.addEventListener('click', () => {
  displayedCount += initialCount;
  const query = searchInput.value;
  const filteredSigns = searchTrafficSigns(query);
  displayTrafficSigns(filteredSigns);
});

displayTrafficSigns(trafic_signs);
