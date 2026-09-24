"use client";

import { useEffect, useState } from "react";

export default function Home() {
  const [status, setStatus] = useState("Checking...");

  useEffect(() => {
    fetch("http://127.0.0.1:8000/health")
      .then((response) => response.json())
      .then((data) => {
        setStatus(data.status === "healthy" ? "Connected" : "Unavailable");
      })
      .catch(() => {
        setStatus("Not connected");
      });
  }, []);

  return (
    <main style={{ padding: "40px", fontFamily: "Arial, sans-serif" }}>
      <h1>COMP902 Project Prototype</h1>

      <p>
        Initial development environment for the COMP902 specialised project.
      </p>

      <div style={{ marginTop: "30px" }}>
        <strong>Backend status:</strong> {status}
      </div>
    </main>
  );
}