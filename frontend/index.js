const form = document.forms.range;
form.addEventListener('submit', async (event) => {
	event.preventDefault();
	const spinner = document.getElementById('spinner');
	spinner.style.display = 'block';
	spinner.src = 'spinner.svg';

	const title = document.getElementById('title');
	const detection = document.getElementById('detection');
	const match = document.getElementById('match');
	title.style.display = 'none';
	match.style.display = 'none';
	detection.style.display = 'none';

	const folder = form.folder.value;
	const start = form.start.value;
	const end = form.end.value;

	const areaUpperTh = form.areaUpperTh.value;
	const areaLowerTh = form.areaLowerTh.value;

	const extendUpperTh = form.extendUpperTh.value;
	const extendLowerTh = form.extendLowerTh.value;

	const majorAxisUpperTh = form.majorAxisUpperTh.value;
	const majorAxisLowerTh = form.majorAxisLowerTh.value;

	const eccentricityUpperTh = form.eccentricityUpperTh.value;
	const eccentricityLowerTh = form.eccentricityLowerTh.value;

	const res = await fetch('http://127.0.0.1:5000', {
		method: 'POST',
		headers: {
			'Content-Type': 'application/json'
		},
		body: JSON.stringify({
			folder,
			start,
			end,
			areaUpperTh,
			areaLowerTh,
			extendUpperTh,
			extendLowerTh,
			majorAxisUpperTh,
			majorAxisLowerTh,
			eccentricityUpperTh,
			eccentricityLowerTh,
		}),
	});
	const data = await res.json();
	spinner.style.display = 'none';

	if (!res.ok) {
		const snackbar = document.getElementById('alert');
		snackbar.style.display = 'flex';
		snackbar.innerText = data.message;
		setTimeout(() => {
			snackbar.style.display = 'none';
		}, 2000);
	} else {
		title.style.display = 'block';
		detection.src = '../candidate_detection.jpg';
		detection.style.display = 'block';
		match.src = '../region_growing.jpg';
		match.style.display = 'block';
	}
});