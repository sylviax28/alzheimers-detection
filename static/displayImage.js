document.getElementById("start-btn").addEventListener("click", async () => {
    const input = document.getElementById("input-image");
    const file = input.files[0];

    if (!file) {
        alert("Please select an image first");
        return;
    }

    //Display the image
    const reader = new FileReader();
    reader.onload = function(e) {
        document.getElementById("display-image").style.backgroundImage = `url(${e.target.result})`;
    }
    reader.readAsDataURL(file);

\    const resultPara = document.getElementById("result");
    //resultPara.innerText = "Loading...";

    //Send user input file to Flask
    const formData = new FormData();
    formData.append("file", file);

    const response = await fetch("/api/run_model", {
        method: "POST",
        body: formData
    });

    const data = await response.json();

    //Updated prediction
    //Check if match is mild or moderate?
    if (data.prediction === "match") {
        resultPara.innerText = "Detected✔️";
        
        resultPara.style.color = "green";
    } else {
        resultPara.innerText = "Not Detected❌";
        resultPara.style.color = "red";

    }
});
