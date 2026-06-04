function drawGauge(score) {

    const canvas = document.getElementById("gauge");
    const ctx = canvas.getContext("2d");

    const width = canvas.width;
    const height = canvas.height;

    ctx.clearRect(0, 0, width, height);

    let centerX = width / 2;
    let centerY = height - 20;
    let radius = 150;

    // Background arc
    ctx.beginPath();
    ctx.arc(centerX, centerY, radius, Math.PI, 2 * Math.PI);
    ctx.lineWidth = 20;
    ctx.strokeStyle = "#ddd";
    ctx.stroke();

    // Color logic
    let color = "#4CAF50";
    if(score > 60) color = "#ff9800";
    if(score > 80) color = "#f44336";

    let endAngle = Math.PI + (score / 100) * Math.PI;

    // Foreground arc
    ctx.beginPath();
    ctx.arc(centerX, centerY, radius, Math.PI, endAngle);
    ctx.lineWidth = 20;
    ctx.strokeStyle = color;
    ctx.stroke();

    // Needle
    let angle = endAngle;
    let needleLength = 120;

    let x = centerX + needleLength * Math.cos(angle);
    let y = centerY + needleLength * Math.sin(angle);

    ctx.beginPath();
    ctx.moveTo(centerX, centerY);
    ctx.lineTo(x, y);
    ctx.lineWidth = 4;
    ctx.strokeStyle = "#000";
    ctx.stroke();

    // Center circle
    ctx.beginPath();
    ctx.arc(centerX, centerY, 6, 0, 2*Math.PI);
    ctx.fillStyle = "#000";
    ctx.fill();

    // Score text
    ctx.font = "bold 32px Arial";
    ctx.fillStyle = "#333";
    ctx.textAlign = "center";
    ctx.fillText(score + "%", centerX, centerY - 50);
}