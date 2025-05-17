const express = require("express");
const bodyParser = require("body-parser");
const { exec } = require("child_process");
const fs = require("fs");
const path = require("path");
const app = express();
const port = process.env.PORT || 3000;

// Middleware
app.use(bodyParser.json({ limit: "50mb" }));
app.use(express.static("public"));

// Routes
app.post("/api/process-transcript-url", (req, res) => {
  const { url } = req.body;
  if (!url) {
    return res.status(400).json({
      success: false,
      error: "No URL provided",
    });
  }

  // Create a temporary data.csv file with the URL
  const csvContent = "Transcription URL\n" + url;
  fs.writeFileSync("data.csv", csvContent);

  // Execute the transcript.py script
  exec("python transcript_url.py", (error, stdout, stderr) => {
    if (error) {
      console.error(`Error executing transcript.py: ${error}`);
      return res.status(500).json({
        success: false,
        error: `Error processing transcript: ${error.message}`,
      });
    }

    console.log(`transcript.py output: ${stdout}`);

    try {
      // Parse the JSON response from the Python script
      let jsonOutput;
      try {
        // Look for JSON output in stdout
        const jsonMatch = stdout.match(/(\{.*\})/s);
        if (jsonMatch) {
          jsonOutput = JSON.parse(jsonMatch[1]);
        } else {
          throw new Error("No valid JSON found in Python output");
        }
      } catch (parseError) {
        console.error(`Error parsing JSON output: ${parseError}`);
        // Fallback to previous approach
        let reportContent;
        if (fs.existsSync("all_reports.md")) {
          reportContent = fs.readFileSync("all_reports.md", "utf8");
        } else {
          const reportMatch = stdout.match(
            /Report Content\s+(# Solar Call Analysis Report[\s\S]+)/
          );
          reportContent = reportMatch
            ? reportMatch[1]
            : "No report content found";
        }

        jsonOutput = {
          success: true,
          report: reportContent,
          charts: [],
        };
      }

      // Process chart files
      const charts = [];
      const analysisFiles = fs
        .readdirSync(".")
        .filter(
          (file) =>
            file.match(/transcript_\d+_analysis\.png/) ||
            file === "call_analysis.png" ||
            file === "default_visualization.png"
        );

      if (analysisFiles.length > 0) {
        // Create public/images directory if it doesn't exist
        if (!fs.existsSync("public/images")) {
          fs.mkdirSync("public/images", { recursive: true });
        }

        analysisFiles.forEach((file) => {
          const publicPath = path.join("public/images", file);
          fs.copyFileSync(file, publicPath);
          charts.push({
            name: file,
            path: "/images/" + file,
          });
        });
      } else {
        // Create a default empty chart file if none exists
        const defaultChartPath = path.join("public/images", "empty_chart.png");

        // Check if we need to create the empty chart
        if (!fs.existsSync(defaultChartPath)) {
          // Create public/images directory if it doesn't exist
          if (!fs.existsSync("public/images")) {
            fs.mkdirSync("public/images", { recursive: true });
          }

          // Copy a placeholder image or create an empty file
          try {
            // Try to find a default image in the project
            const defaultImage = fs.existsSync("default.png")
              ? "default.png"
              : fs.existsSync("public/default.png")
              ? "public/default.png"
              : null;

            if (defaultImage) {
              fs.copyFileSync(defaultImage, defaultChartPath);
            } else {
              // Create an empty file as last resort
              fs.writeFileSync(defaultChartPath, "");
            }
          } catch (e) {
            console.error("Error creating default chart:", e);
          }
        }

        charts.push({
          name: "empty_chart.png",
          path: "/images/empty_chart.png",
        });
      }

      // Add charts to the response if not already included
      if (!jsonOutput.charts || jsonOutput.charts.length === 0) {
        jsonOutput.charts = charts;
      } else {
        // Update paths to be web-accessible
        jsonOutput.charts = jsonOutput.charts.map((chart) => ({
          name: chart.name,
          path: "/images/" + chart.name,
        }));
      }

      return res.json(jsonOutput);
    } catch (readError) {
      console.error(`Error processing output: ${readError}`);
      return res.status(500).json({
        success: false,
        error: `Error processing analysis results: ${readError.message}`,
      });
    }
  });
});

app.post("/api/process-transcript-json", (req, res) => {
  const { json } = req.body;

  if (!json) {
    return res.status(400).json({
      success: false,
      error: "No JSON data provided",
    });
  }

  // Create a temporary file to pass the JSON directly to Python
  const tempJsonString = JSON.stringify(json);

  // Execute the transcript_json.py script and pass the JSON data directly via stdin
  const pythonProcess = exec(
    "python transcript_json.py",
    { maxBuffer: 1024 * 1024 * 10 }, // Increase buffer size for large transcripts
    (error, stdout, stderr) => {
      if (error) {
        console.error(`Error executing transcript_json.py: ${error}`);
        return res.status(500).json({
          success: false,
          error: `Error processing transcript: ${error.message}`,
        });
      }

      console.log(`transcript_json.py output: ${stdout}`);

      try {
        // Parse the JSON response from the Python script
        let jsonOutput;
        try {
          // Look for JSON output in stdout
          const jsonMatch = stdout.match(/(\{.*\})/s);
          if (jsonMatch) {
            jsonOutput = JSON.parse(jsonMatch[1]);
          } else {
            throw new Error("No valid JSON found in Python output");
          }
        } catch (parseError) {
          console.error(`Error parsing JSON output: ${parseError}`);
          // Fallback to previous approach
          let reportContent;
          if (fs.existsSync("all_reports.md")) {
            reportContent = fs.readFileSync("all_reports.md", "utf8");
          } else {
            const reportMatch = stdout.match(
              /Report Content\s+(# Solar Call Analysis Report[\s\S]+)/
            );
            reportContent = reportMatch
              ? reportMatch[1]
              : "No report content found";
          }

          jsonOutput = {
            success: true,
            report: reportContent,
            charts: [],
          };
        }

        // Process chart files
        const charts = [];
        const analysisFiles = fs
          .readdirSync(".")
          .filter(
            (file) =>
              file.match(/transcript_\d+_analysis\.png/) ||
              file === "call_analysis.png" ||
              file === "default_visualization.png"
          );

        if (analysisFiles.length > 0) {
          // Create public/images directory if it doesn't exist
          if (!fs.existsSync("public/images")) {
            fs.mkdirSync("public/images", { recursive: true });
          }

          analysisFiles.forEach((file) => {
            const publicPath = path.join("public/images", file);
            fs.copyFileSync(file, publicPath);
            charts.push({
              name: file,
              path: "/images/" + file,
            });
          });
        } else {
          // Create a default empty chart file if none exists
          const defaultChartPath = path.join(
            "public/images",
            "empty_chart.png"
          );

          // Ensure the default chart exists (same logic as in other route)
          if (!fs.existsSync(defaultChartPath)) {
            if (!fs.existsSync("public/images")) {
              fs.mkdirSync("public/images", { recursive: true });
            }

            try {
              const defaultImage = fs.existsSync("default.png")
                ? "default.png"
                : fs.existsSync("public/default.png")
                ? "public/default.png"
                : null;

              if (defaultImage) {
                fs.copyFileSync(defaultImage, defaultChartPath);
              } else {
                fs.writeFileSync(defaultChartPath, "");
              }
            } catch (e) {
              console.error("Error creating default chart:", e);
            }
          }

          charts.push({
            name: "empty_chart.png",
            path: "/images/empty_chart.png",
          });
        }

        // Add charts to the response if not already included
        if (!jsonOutput.charts || jsonOutput.charts.length === 0) {
          jsonOutput.charts = charts;
        } else {
          // Update paths to be web-accessible
          jsonOutput.charts = jsonOutput.charts.map((chart) => ({
            name: chart.name,
            path: "/images/" + chart.name,
          }));
        }

        return res.json(jsonOutput);
      } catch (readError) {
        console.error(`Error processing output: ${readError}`);
        return res.status(500).json({
          success: false,
          error: `Error processing analysis results: ${readError.message}`,
        });
      }
    }
  );

  // Write the JSON data to the Python process stdin
  pythonProcess.stdin.write(tempJsonString);
  pythonProcess.stdin.end();
});

// Start the server
app.listen(port, () => {
  console.log(`Server running on port ${port}`);
  console.log(`Access the dashboard at http://localhost:${port}`);
});
