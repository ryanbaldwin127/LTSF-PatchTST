const fs = require('node:fs/promises');

// Don't care about leap years.
const DAYS_PER_MONTH = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];

function wait(ms) {
	return new Promise(resolve => {
		setTimeout(resolve, ms);
	})
}

function range(start, n) {
	return Array.from({ length: n }, (_, i) => i + start);
}

async function fetchDay(year, month, day) {
	const LINE_SEP = "<br>";
	const HEADER_ROWS = 3;

	const response = await fetch("https://climate.colostate.edu/~autowx/fclwx_results.php", {
		method: "POST",
		headers: {
			"content-type": "application/x-www-form-urlencoded",
		},
		body: `year%5B%5D=${year}&month%5B%5D=${month}&day%5B%5D=${day}&table=&table%5B%5D=Hourly&format=csv`,
	});

	const dataRows = (await response.text()).split(LINE_SEP).slice(HEADER_ROWS);
	return dataRows
		.map(r => r.split(",").map(x => x.replace(/"/g, "")).join())
		.filter(x => !!x);
}

async function fetchCsuWeather(startYear, numYears) {
	console.log(`Start year = ${startYear}. ${numYears} years of data.`)

	const dates = range(startYear, numYears)
		.flatMap(year => {
			return range(0, 12).flatMap(month => {
				return range(0, DAYS_PER_MONTH[month])
					.map(day => [year, month + 1, day + 1])
			})
		});

	console.log(`Fetching ${dates.length} dates.`);

	const allRecords = [];
	for (const [i, date] of dates.entries()) {
		const dailyRows = await fetchDay(...date);
		console.log(`${i}: Read ${dailyRows.length} rows.`)
		allRecords.push(...dailyRows);

		await wait(400);
	}

	await fs.writeFile(
		`csu_weather_${startYear}_${startYear + numYears - 1}_dates.csv`,
		allRecords.join("\n"),
		"utf8"
	);

	// Remove first two columns (date and time).
	await fs.writeFile(
		`csu_weather_${startYear}_${startYear + numYears - 1}.csv`,
		allRecords.map(row => row.split(",").slice(2).join(",")).join("\n"),
		"utf8"
	);
}

fetchCsuWeather(2020, 5).catch((err) => {
	console.error('Failed to save CSV:', err);
	process.exit(1);
});
