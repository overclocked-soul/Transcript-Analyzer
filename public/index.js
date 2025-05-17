document.addEventListener("DOMContentLoaded", () => {
  // DOM Elements
  const inputToggleLabels = document.querySelectorAll(".toggle-label input");
  const urlInputContainer = document.getElementById("url-input-container");
  const jsonInputContainer = document.getElementById("json-input-container");
  const transcriptUrl = document.getElementById("transcript-url");
  const transcriptJson = document.getElementById("transcript-json");
  const analyzeBtn = document.getElementById("analyze-btn");
  const loader = document.getElementById("loader");
  const analysisResults = document.getElementById("analysis-results");
  const errorMessage = document.getElementById("error-message");
  const errorText = document.getElementById("error-text");

  // Metrics Elements
  const totalMessages = document.getElementById("total-messages");
  const agentMessages = document.getElementById("agent-messages");
  const customerMessages = document.getElementById("customer-messages");
  const avgAgentWords = document.getElementById("avg-agent-words");
  const avgCustomerWords = document.getElementById("avg-customer-words");
  const callDate = document.getElementById("call-date");

  // Tab Elements
  const tabs = document.querySelectorAll(".tab");
  const summaryTab = document.getElementById("summary-tab");
  const scriptTab = document.getElementById("script-tab");
  const objectionsTab = document.getElementById("objections-tab");
  const recsTab = document.getElementById("recs-tab");

  // Content Elements
  const summaryContent = document.getElementById("summary-content");
  const scriptContent = document.getElementById("script-content");
  const objectionsContent = document.getElementById("objections-content");
  const recsContent = document.getElementById("recs-content");
  //   const mainVisualization = document.getElementById("main-visualization");
  const additionalCharts = document.getElementById("additional-charts");
  const searchBox = document.getElementById("search-box");

  // Toggle between URL and JSON input
  inputToggleLabels.forEach((input) => {
    input.addEventListener("change", (e) => {
      if (e.target.value === "url") {
        urlInputContainer.classList.remove("hidden");
        jsonInputContainer.classList.add("hidden");
      } else {
        urlInputContainer.classList.add("hidden");
        jsonInputContainer.classList.remove("hidden");
      }
    });
  });

  // Tab switching functionality
  tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      // Remove active class from all tabs
      tabs.forEach((t) => t.classList.remove("active"));

      // Add active class to clicked tab
      tab.classList.add("active");

      // Hide all tab content
      summaryTab.classList.add("hidden");
      scriptTab.classList.add("hidden");
      objectionsTab.classList.add("hidden");
      recsTab.classList.add("hidden");

      // Show selected tab content
      const tabName = tab.getAttribute("data-tab");
      if (tabName === "summary") {
        summaryTab.classList.remove("hidden");
      } else if (tabName === "script") {
        scriptTab.classList.remove("hidden");
      } else if (tabName === "objections") {
        objectionsTab.classList.remove("hidden");
      } else if (tabName === "recs") {
        recsTab.classList.remove("hidden");
      }
    });
  });

  // Search functionality
  searchBox.addEventListener("input", () => {
    const searchTerm = searchBox.value.toLowerCase();
    highlightSearch(searchTerm);
  });

  // Highlight search terms in the transcript analysis
  function highlightSearch(searchTerm) {
    if (!searchTerm) {
      // If search term is empty, remove all highlights
      document.querySelectorAll(".highlight").forEach((el) => {
        el.outerHTML = el.innerHTML;
      });
      return;
    }

    const contentElements = [
      summaryContent,
      scriptContent,
      objectionsContent,
      recsContent,
    ];

    contentElements.forEach((element) => {
      if (!element) return;

      // Get the content and replace highlighted parts
      let content = element.innerHTML;

      // Remove existing highlights first
      content = content.replace(
        /<span class="highlight">([^<]+)<\/span>/g,
        "$1"
      );

      // Apply new highlights
      if (searchTerm) {
        const regex = new RegExp(`(${searchTerm})`, "gi");
        content = content.replace(regex, '<span class="highlight">$1</span>');
      }

      element.innerHTML = content;
    });
  }

  // Analyze button click handler
  analyzeBtn.addEventListener("click", () => {
    // Hide previous results and errors
    analysisResults.classList.add("hidden");
    errorMessage.classList.add("hidden");

    // Show loader
    loader.classList.remove("hidden");

    // Check which input method is selected
    const selectedInputType = document.querySelector(
      ".toggle-label input:checked"
    ).value;

    if (selectedInputType === "url") {
      // Process URL input
      if (!transcriptUrl.value.trim()) {
        showError("Please enter a transcript URL");
        loader.classList.add("hidden");
        return;
      }

      // Call API to process URL
      fetch("/api/process-transcript-url", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ url: transcriptUrl.value.trim() }),
      })
        .then((response) => response.json())
        .then((data) => processAnalysisResponse(data))
        .catch((error) => {
          showError("Error processing transcript URL: " + error.message);
          console.error("Error:", error);
        });
    } else {
      // Process JSON input
      if (!transcriptJson.value.trim()) {
        showError("Please enter JSON transcript content");
        loader.classList.add("hidden");
        return;
      }

      // Try to parse JSON to validate it
      let jsonData;
      try {
        jsonData = JSON.parse(transcriptJson.value.trim());
      } catch (e) {
        showError("Invalid JSON format: " + e.message);
        loader.classList.add("hidden");
        return;
      }

      // Call API to process JSON
      fetch("/api/process-transcript-json", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ json: jsonData }),
      })
        .then((response) => response.json())
        .then((data) => processAnalysisResponse(data))
        .catch((error) => {
          showError("Error processing transcript JSON: " + error.message);
          console.error("Error:", error);
        });
    }
  });

  // Process the analysis response from the server
  function processAnalysisResponse(data) {
    // Hide loader
    loader.classList.add("hidden");

    // Check if data exists
    if (!data) {
      showError("No response data received from server");
      return;
    }

    // Check if the API call was successful
    if (!data.success) {
      showError(data.error || "Unknown error occurred");
      return;
    }

    try {
      // Parse the report content
      const reportContent = data.report || "";

      // Make sure we have report content
      if (!reportContent.trim()) {
        showError("Empty report received from server");
        return;
      }

      // Extract metrics from the report
      extractAndDisplayMetrics(reportContent);

      // Display the report content in different tabs
      displayReportContent(reportContent);

      // Display visualizations - ensure the charts property exists
      displayVisualizations(Array.isArray(data.charts) ? data.charts : []);

      // Show the analysis results section
      analysisResults.classList.remove("hidden");
    } catch (error) {
      // Catch any errors during processing
      console.error("Error processing analysis data:", error);
      showError("Error processing analysis results: " + error.message);
    }
  }

  // Extract metrics from the report content
  function extractAndDisplayMetrics(reportContent) {
    // Default values
    let metrics = {
      totalMessages: "-",
      agentMessages: "-",
      customerMessages: "-",
      avgAgentWords: "-",
      avgCustomerWords: "-",
      callDate: "-",
    };

    // Try to extract metrics from report content
    try {
      // Total messages
      const totalMatch = reportContent.match(/Total messages:\s*(\d+)/i);
      if (totalMatch) metrics.totalMessages = totalMatch[1];

      // Agent messages
      const agentMatch = reportContent.match(/Agent messages:\s*(\d+)/i);
      if (agentMatch) metrics.agentMessages = agentMatch[1];

      // Customer messages
      const customerMatch = reportContent.match(/Customer messages:\s*(\d+)/i);
      if (customerMatch) metrics.customerMessages = customerMatch[1];

      // Average agent words
      const agentWordsMatch = reportContent.match(
        /Average agent message length:\s*([\d\.]+)/i
      );
      if (agentWordsMatch) metrics.avgAgentWords = agentWordsMatch[1];

      // Average customer words
      const customerWordsMatch = reportContent.match(
        /Average customer message length:\s*([\d\.]+)/i
      );
      if (customerWordsMatch) metrics.avgCustomerWords = customerWordsMatch[1];

      // Call date
      const dateMatch = reportContent.match(
        /Call date:\s*(\d{4}-\d{2}-\d{2})/i
      );
      if (dateMatch) metrics.callDate = dateMatch[1];

      console.log("Extracted metrics:", metrics); // Debug log
    } catch (e) {
      console.error("Error extracting metrics:", e);
    }

    // Update the DOM with extracted metrics
    totalMessages.textContent = metrics.totalMessages;
    agentMessages.textContent = metrics.agentMessages;
    customerMessages.textContent = metrics.customerMessages;
    avgAgentWords.textContent = metrics.avgAgentWords;
    avgCustomerWords.textContent = metrics.avgCustomerWords;
    callDate.textContent = metrics.callDate;
  }

  // Display the report content in different tabs
  function displayReportContent(reportContent) {
    // Use marked library to convert markdown to HTML
    const htmlContent = marked.parse(reportContent);

    // Create a temporary element to parse the HTML content
    const tempElement = document.createElement("div");
    tempElement.innerHTML = htmlContent;

    // Extract content for each tab

    // Summary tab - extract "Call Summary" section
    const summarySection = extractSection(
      tempElement,
      "Call Summary",
      "Script Adherence"
    );
    summaryContent.innerHTML =
      summarySection || "<p>No summary information available.</p>";

    // Script adherence tab - extract both "Script Adherence Summary" and "Script Flow Analysis"
    const scriptSection = extractSection(
      tempElement,
      "Script Adherence",
      "Objection Handling"
    );
    scriptContent.innerHTML =
      scriptSection || "<p>No script adherence information available.</p>";

    // Objections tab - extract "Objection Handling Analysis" and "Communication Quality Issues"
    const objectionsSection = extractSection(
      tempElement,
      "Objection Handling",
      "Recommended Improvements"
    );
    objectionsContent.innerHTML =
      objectionsSection ||
      "<p>No objection handling information available.</p>";

    // Recommendations tab - extract both "Recommended Improvements" and "Issue Summary"
    const recsSection = extractSection(
      tempElement,
      "Recommended Improvements",
      null
    );
    recsContent.innerHTML =
      recsSection || "<p>No recommendations available.</p>";

    // Add markdown-content class to all content divs for consistent styling
    [summaryContent, scriptContent, objectionsContent, recsContent].forEach(
      (element) => {
        element.classList.add("markdown-content");
      }
    );
  }

  // Helper function to extract a section from the HTML content
  function extractSection(element, startHeading, endHeading) {
    let startFound = false;
    let content = "";
    let currentElement = element.firstChild;

    while (currentElement) {
      // Check if this is the start heading - look for partial matches in headings of any level
      if (
        !startFound &&
        currentElement.tagName &&
        currentElement.tagName.match(/^H[1-6]$/i) &&
        currentElement.textContent.includes(startHeading)
      ) {
        startFound = true;
        content += currentElement.outerHTML;
      }
      // If we're in the section, add content until we reach the end heading
      else if (startFound) {
        // Check if this is the end heading - look for partial matches in headings of any level
        if (
          endHeading &&
          currentElement.tagName &&
          currentElement.tagName.match(/^H[1-6]$/i) &&
          currentElement.textContent.includes(endHeading)
        ) {
          break;
        }
        // Add this element to our content
        content += currentElement.outerHTML || "";
      }

      currentElement = currentElement.nextSibling;
    }

    return content;
  }

  // Display visualizations
  function displayVisualizations(charts) {
    // // First, validate the charts parameter
    // if (!charts || !Array.isArray(charts) || charts.length === 0) {
    //   mainVisualization.innerHTML =
    //     "<p>No visualizations available for this transcript.</p>";
    //   additionalCharts.innerHTML = "";
    //   return;
    // }
    // // Make sure the first chart has a valid path
    // if (!charts[0] || !charts[0].path) {
    //   mainVisualization.innerHTML = "<p>Visualization data is incomplete.</p>";
    //   additionalCharts.innerHTML = "";
    //   return;
    // }
    // // Display the first chart as the main visualization
    // mainVisualization.innerHTML = `<img src="${charts[0].path}" alt="Call Analysis Visualization">`;
    // // Display additional charts
    // if (charts.length > 1) {
    //   let validAdditionalCharts = charts
    //     .slice(1)
    //     .filter((chart) => chart && chart.path);
    //   if (validAdditionalCharts.length > 0) {
    //     let additionalChartsHTML = "";
    //     for (let i = 0; i < validAdditionalCharts.length; i++) {
    //       additionalChartsHTML += `<img src="${
    //         validAdditionalCharts[i].path
    //       }" alt="Additional Chart ${i + 1}">`;
    //     }
    //     additionalCharts.innerHTML = additionalChartsHTML;
    //   } else {
    //     additionalCharts.innerHTML =
    //       "<p>No additional charts available for this transcript.</p>";
    //   }
    // } else {
    //   additionalCharts.innerHTML =
    //     "<p>No additional charts available for this transcript.</p>";
    // }
  }

  // Show error message
  function showError(message) {
    errorText.textContent = message;
    errorMessage.classList.remove("hidden");
    analysisResults.classList.add("hidden");
  }

  function ensureDefaultChartExists() {
    const defaultChartPath = path.join(
      __dirname,
      "public/images/empty_chart.png"
    );
    const defaultChartDir = path.dirname(defaultChartPath);

    // Create directory if it doesn't exist
    if (!fs.existsSync(defaultChartDir)) {
      fs.mkdirSync(defaultChartDir, { recursive: true });
    }

    // Create an empty chart if it doesn't exist
    if (!fs.existsSync(defaultChartPath)) {
      try {
        // Try to find a default image in the project
        const defaultImage = fs.existsSync(path.join(__dirname, "default.png"))
          ? path.join(__dirname, "default.png")
          : null;

        if (defaultImage) {
          fs.copyFileSync(defaultImage, defaultChartPath);
          console.log("Created empty_chart.png from default.png");
        } else {
          // Create a small 1x1 empty PNG
          const emptyPng = Buffer.from([
            0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00,
            0x0d, 0x49, 0x48, 0x44, 0x52, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00,
            0x00, 0x01, 0x08, 0x06, 0x00, 0x00, 0x00, 0x1f, 0x15, 0xc4, 0x89,
            0x00, 0x00, 0x00, 0x0a, 0x49, 0x44, 0x41, 0x54, 0x78, 0x9c, 0x63,
            0x00, 0x01, 0x00, 0x00, 0x05, 0x00, 0x01, 0x0d, 0x0a, 0x2d, 0xb4,
            0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4e, 0x44, 0xae, 0x42, 0x60,
            0x82,
          ]);
          fs.writeFileSync(defaultChartPath, emptyPng);
          console.log("Created empty 1x1 PNG for empty_chart.png");
        }
      } catch (e) {
        console.error("Error creating default chart:", e);
      }
    }
  }

  ensureDefaultChartExists();
});
