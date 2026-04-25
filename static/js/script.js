function createBarChart(id, labels, data, label, horizontal = false) {
    const el = document.getElementById(id);
    if (!el) return;
    new Chart(el, {
        type: "bar",
        data: {
            labels: labels,
            datasets: [{
                label: label,
                data: data,
                borderWidth: 1,
                backgroundColor: "rgba(13, 110, 253, 0.55)",
                borderColor: "rgba(13, 110, 253, 1)"
            }]
        },
        options: {
            indexAxis: horizontal ? "y" : "x",
            responsive: true,
            plugins: { legend: { display: true } },
            scales: {
                y: { beginAtZero: true }
            }
        }
    });
}

function createLineChart(id, labels, data, label) {
    const el = document.getElementById(id);
    if (!el) return;
    new Chart(el, {
        type: "line",
        data: {
            labels: labels,
            datasets: [{
                label: label,
                data: data,
                fill: false,
                tension: 0.2,
                borderWidth: 3,
                backgroundColor: "rgba(25, 135, 84, 0.35)",
                borderColor: "rgba(25, 135, 84, 1)"
            }]
        },
        options: {
            responsive: true,
            scales: {
                y: { beginAtZero: true, max: 1 }
            }
        }
    });
}

document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("searchForm");
    const spinner = document.getElementById("loadingSpinner");
    if (form && spinner) {
        form.addEventListener("submit", () => {
            spinner.classList.remove("d-none");
        });
    }

    const datasetSearch = document.getElementById("datasetSearch");
    const datasetTable = document.getElementById("datasetTable");
    if (datasetSearch && datasetTable) {
        datasetSearch.addEventListener("keyup", () => {
            const search = datasetSearch.value.toLowerCase();
            const rows = datasetTable.querySelectorAll("tbody tr");
            rows.forEach((row) => {
                row.style.display = row.innerText.toLowerCase().includes(search) ? "" : "none";
            });
        });
    }

    if (window.chartPayload) {
        const payload = window.chartPayload;
        createBarChart("tfChart", payload.labels, payload.tf_sums, "TF (sum per top document)");
        createBarChart("tfidfChart", payload.labels, payload.tfidf_sums, "TF-IDF (sum per top document)");
        createBarChart("rankingChart", payload.labels, payload.cosine_values, "Cosine Similarity Ranking", true);
        createLineChart("precisionChart", payload.precision_k, payload.precision_values, "Precision@K");
        createLineChart("precisionOnlyChart", payload.precision_k, payload.precision_values, "Precision@K");
    }
});
