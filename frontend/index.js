const intermediate = document.getElementById('intermediate');
const button = document.getElementById('button');
button.onclick = () => {
	intermediate.style.display = 'flex';
	const detection = document.getElementById('detection');
	const match = document.getElementById('match');
	match.src = '../region_growing.jpg';
	detection.src = '../candidate_detection.jpg';
};

const area = document.getElementById('area');
const extent = document.getElementById('extent');
const majorAxis = document.getElementById('majorAxis');
const eccentricity = document.getElementById('eccentricity');
const histogarms = [area, extent, majorAxis, eccentricity];
const tracking = document.getElementById('tracking');
const trackingButton = [tracking, button];
const spinner = document.getElementById('spinner');
const items = [...histogarms, ...trackingButton, intermediate];
const snackbar = document.getElementById('alert');

const buttonClick = () => {
	spinner.style.display = 'block';
	spinner.src = 'spinner.svg';
	document.querySelectorAll('.button').forEach((button) => button.setAttribute('disabled', ''));
	const folder = form.folder.value;
	const frames = form.frames.value ? parseInt(form.frames.value) : 1;
	const start = form.start.value ? form.start.value : 1;
	const end = form.end.value;
	return [folder, frames, start, end];
};

const resolve = () => {
	spinner.style.display = 'none';
	document.querySelectorAll('.button').forEach((button) => button.removeAttribute('disabled'));
};

const form = document.forms.range;
document.getElementById('calibration').addEventListener('click', async (event) => {
	event.preventDefault();
	const [folder, frames, start, end] = buttonClick();
	items.forEach((histogarm) => histogarm.style.display = 'none');

	const res = await fetch('http://127.0.0.1:5000/calibration', {
		method: 'POST',
		headers: {
			'Content-Type': 'application/json'
		},
		body: JSON.stringify({
			folder,
			frames,
			start,
			end,
		}),
	});
	const data = await res.json();
	resolve();
	if (!res.ok) {
		snackbar.style.display = 'flex';
		snackbar.innerText = data.message;
		setTimeout(() => {
			snackbar.style.display = 'none';
		}, 2000);
	} else {
		histogarms.forEach((histogarm) => {
			histogarm.style.display = 'block'
			histogarm.src = `../${histogarm.id}.jpg`;
		});
	}
});

document.getElementById('track').addEventListener('click', async (event) => {
	event.preventDefault();
	const [folder, frames, start, end] = buttonClick();
	items.forEach((histogarm) => histogarm.style.display = 'none');

	const areaUpperTh = form.areaUpperTh.value;
	const areaLowerTh = form.areaLowerTh.value;

	const extentUpperTh = form.extentUpperTh.value;
	const extentLowerTh = form.extentLowerTh.value;

	const majorAxisUpperTh = form.majorAxisUpperTh.value;
	const majorAxisLowerTh = form.majorAxisLowerTh.value;

	const eccentricityUpperTh = form.eccentricityUpperTh.value;
	const eccentricityLowerTh = form.eccentricityLowerTh.value;

	const res = await fetch('http://127.0.0.1:5000/track', {
		method: 'POST',
		headers: {
			'Content-Type': 'application/json'
		},
		body: JSON.stringify({
			folder,
			frames,
			start,
			end,
			areaUpperTh,
			areaLowerTh,
			extentUpperTh,
			extentLowerTh,
			majorAxisUpperTh,
			majorAxisLowerTh,
			eccentricityUpperTh,
			eccentricityLowerTh,
		}),
	});
	const data = await res.json();
	resolve();
	if (!res.ok) {
		snackbar.style.display = 'flex';
		snackbar.innerText = data.message;
		setTimeout(() => {
			snackbar.style.display = 'none';
		}, 2000);
	} else {
		trackingButton.forEach((image) => image.style.display = 'block');
		let frame = parseInt(start) + frames;
		tracking.src = `../${frame}.jpg`;
		const interval = setInterval(() => {
			const finishTrackingDisplay = () => {
				clearInterval(interval);
				tracking.src = '../graph.jpg';
			};
			tracking.src = `../${frame += frames}.jpg`;
			tracking.onerror = () => {
				tracking.src = `../${frame -= frames}.jpg`;
				finishTrackingDisplay();
			};
			if (end && frame > end - frames) finishTrackingDisplay();
		}, 500);
	}
});