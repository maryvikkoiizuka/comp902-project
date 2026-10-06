"use client";

import { useEffect, useState } from "react";
import type { CSSProperties, FormEvent } from "react";

const API_BASE = "http://127.0.0.1:8000";
const INITIAL_DOCUMENT_ID = "bccb14bb-f26f-4996-924a-e182105acf11";

type CoverageStatus =
  | "covered"
  | "drafts_pending_review"
  | "rejected_only"
  | "no_tests";

type RequirementCoverage = {
  document_id: string;
  requirement_id: string;
  text: string;
  source_line: number;
  coverage_status: CoverageStatus;
  covered: boolean;
  test_case_count: number;
  approved_test_count: number;
  draft_test_count: number;
  rejected_test_count: number;
  test_case_ids: string[];
};

type CoverageResponse = {
  document_id: string;
  title: string;
  coverage_basis: string;
  coverage_scope: string;
  requirement_count: number;
  covered_requirement_count: number;
  uncovered_requirement_count: number;
  coverage_percentage: number | null;
  status_counts: Record<CoverageStatus, number>;
  requirements: RequirementCoverage[];
};

const statusLabels: Record<CoverageStatus, string> = {
  covered: "Approved test available",
  drafts_pending_review: "Drafts awaiting review",
  rejected_only: "Rejected tests only",
  no_tests: "No tests",
};

const statusColors: Record<CoverageStatus, { background: string; color: string }> = {
  covered: { background: "#e7f5ed", color: "#17623b" },
  drafts_pending_review: { background: "#fff3d6", color: "#765300" },
  rejected_only: { background: "#fde9e9", color: "#9d2929" },
  no_tests: { background: "#edf0f4", color: "#465366" },
};

const cellStyle: CSSProperties = {
  padding: "15px 16px",
  borderBottom: "1px solid #e5e9ef",
  textAlign: "left",
  verticalAlign: "top",
};

const buttonStyle: CSSProperties = {
  padding: "11px 18px",
  border: "1px solid #24634f",
  borderRadius: "8px",
  background: "#24634f",
  color: "#ffffff",
  fontWeight: 600,
  cursor: "pointer",
};

export default function Home() {
  const [backendStatus, setBackendStatus] = useState("Checking...");
  const [documentInput, setDocumentInput] = useState(INITIAL_DOCUMENT_ID);
  const [documentId, setDocumentId] = useState(INITIAL_DOCUMENT_ID);
  const [refreshCount, setRefreshCount] = useState(0);
  const [coverage, setCoverage] = useState<CoverageResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    setBackendStatus("Checking...");

    async function checkBackend() {
      try {
        const response = await fetch(`${API_BASE}/health`, {
          signal: controller.signal,
          cache: "no-store",
        });
        if (!response.ok) throw new Error("Health check failed.");
        const data = await response.json();
        if (!controller.signal.aborted) {
          setBackendStatus(data.status === "healthy" ? "Connected" : "Unavailable");
        }
      } catch {
        if (!controller.signal.aborted) setBackendStatus("Not connected");
      }
    }

    void checkBackend();
    return () => controller.abort();
  }, [refreshCount]);

  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    setError(null);
    setCoverage(null);

    async function loadCoverage() {
      try {
        const response = await fetch(
          `${API_BASE}/requirements/documents/${encodeURIComponent(documentId)}/coverage`,
          { signal: controller.signal, cache: "no-store" },
        );

        if (!response.ok) {
          let message = `Unable to load document (HTTP ${response.status}).`;
          try {
            const body = await response.json();
            if (typeof body.detail === "string") message = body.detail;
          } catch {
            // Retain the HTTP message when the error body is not JSON.
          }
          throw new Error(message);
        }

        const data: CoverageResponse = await response.json();
        if (!Array.isArray(data.requirements)) {
          throw new Error("The backend returned an unexpected coverage response.");
        }
        if (!controller.signal.aborted) setCoverage(data);
      } catch (caught) {
        if (!controller.signal.aborted) {
          setError(
            caught instanceof TypeError
              ? "Cannot reach the backend. Check that FastAPI is running and open the frontend at http://localhost:3000."
              : caught instanceof Error
                ? caught.message
                : "Unable to load coverage.",
          );
        }
      } finally {
        if (!controller.signal.aborted) setLoading(false);
      }
    }

    void loadCoverage();
    return () => controller.abort();
  }, [documentId, refreshCount]);

  function handleLoad(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const nextId = documentInput.trim();
    if (!nextId) return;
    setDocumentId(nextId);
    setRefreshCount((count) => count + 1);
  }

  return (
    <main
      style={{
        minHeight: "100vh",
        padding: "32px 20px",
        background: "#f5f7fa",
        color: "#1c293b",
        fontFamily: "Arial, sans-serif",
      }}
    >
      <div style={{ maxWidth: "1150px", margin: "0 auto" }}>
        <header style={{ marginBottom: "26px" }}>
          <p style={{ color: "#24634f", fontSize: "13px", fontWeight: 700, marginBottom: "8px" }}>
            COMP902 PROJECT PROTOTYPE
          </p>
          <h1 style={{ fontSize: "30px", fontWeight: 700, marginBottom: "10px" }}>
            AI-Assisted Test Case Generation
          </h1>
          <p style={{ color: "#596779", lineHeight: 1.6 }}>
            Requirements and approved test-design coverage
          </p>
          <p style={{ marginTop: "12px", fontSize: "14px" }} aria-live="polite">
            <strong>Backend status:</strong> {backendStatus}
          </p>
        </header>

        <form
          onSubmit={handleLoad}
          style={{ padding: "20px", border: "1px solid #e0e6ed", borderRadius: "12px", background: "#ffffff" }}
        >
          <label htmlFor="document-id" style={{ display: "block", fontWeight: 600, marginBottom: "10px" }}>
            Saved document ID
          </label>
          <div style={{ display: "flex", gap: "12px", flexWrap: "wrap" }}>
            <input
              id="document-id"
              value={documentInput}
              onChange={(event) => setDocumentInput(event.target.value)}
              required
              spellCheck={false}
              autoComplete="off"
              style={{ flex: "1 1 300px", minWidth: 0, border: "1px solid #bac5d2", borderRadius: "8px", padding: "11px 12px", background: "#ffffff", color: "#1c293b" }}
            />
            <button
              type="submit"
              disabled={loading || !documentInput.trim()}
              style={{ ...buttonStyle, opacity: loading || !documentInput.trim() ? 0.6 : 1 }}
            >
              {loading ? "Loading..." : "Load / refresh"}
            </button>
          </div>
        </form>

        {loading && <p role="status" style={{ marginTop: "24px" }}>Loading requirements and coverage...</p>}
        {error && (
          <p role="alert" style={{ marginTop: "24px", padding: "16px", borderRadius: "8px", background: "#fde9e9", color: "#9d2929" }}>
            {error}
          </p>
        )}

        {coverage && !loading && (
          <section style={{ marginTop: "28px" }} aria-labelledby="document-title">
            <h2 id="document-title" style={{ fontSize: "22px", fontWeight: 700, marginBottom: "16px" }}>
              {coverage.title}
            </h2>

            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: "14px" }}>
              {[
                { label: "Saved requirements", value: coverage.requirement_count },
                { label: "With approved tests", value: coverage.covered_requirement_count },
                { label: "Without approved tests", value: coverage.uncovered_requirement_count },
                {
                  label: "Approved test-design coverage",
                  value: coverage.coverage_percentage === null ? "N/A" : `${coverage.coverage_percentage.toFixed(2)}%`,
                },
              ].map((card) => (
                <div key={card.label} style={{ padding: "20px", border: "1px solid #e0e6ed", borderRadius: "10px", background: "#ffffff" }}>
                  <p style={{ fontSize: "13px", color: "#596779", marginBottom: "10px", lineHeight: 1.5 }}>{card.label}</p>
                  <p style={{ fontSize: "28px", fontWeight: 700 }}>{card.value}</p>
                </div>
              ))}
            </div>

            <p style={{ margin: "16px 0 24px", fontSize: "13px", color: "#596779", lineHeight: 1.6 }}>
              A requirement counts as covered when it has at least one approved test.
              This does not indicate that tests have been executed or that all scenarios are covered.
            </p>

            <h3 style={{ fontSize: "18px", fontWeight: 700, marginBottom: "12px" }}>Saved requirements</h3>
            {coverage.requirements.length === 0 ? (
              <p style={{ padding: "20px", background: "#ffffff", borderRadius: "10px" }}>
                This document has no saved extracted requirements yet.
              </p>
            ) : (
              <div style={{ overflowX: "auto", border: "1px solid #e0e6ed", borderRadius: "10px", background: "#ffffff" }}>
                <table style={{ width: "100%", minWidth: "850px", borderCollapse: "collapse", fontSize: "14px" }}>
                  <thead style={{ background: "#edf2f7" }}>
                    <tr>
                      {['ID', 'Requirement', 'Source line', 'Coverage status', 'Approved / Draft / Rejected'].map((heading) => (
                        <th key={heading} scope="col" style={{ ...cellStyle, fontWeight: 600 }}>{heading}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {coverage.requirements.map((requirement) => (
                      <tr key={`${requirement.document_id}:${requirement.requirement_id}`}>
                        <th scope="row" style={{ ...cellStyle, whiteSpace: "nowrap", fontWeight: 600 }}>{requirement.requirement_id}</th>
                        <td style={{ ...cellStyle, minWidth: "260px", lineHeight: 1.6 }}>{requirement.text}</td>
                        <td style={cellStyle}>{requirement.source_line}</td>
                        <td style={cellStyle}>
                          <span style={{ ...statusColors[requirement.coverage_status], display: "inline-block", padding: "6px 9px", borderRadius: "6px", fontSize: "12px", whiteSpace: "nowrap" }}>
                            {statusLabels[requirement.coverage_status]}
                          </span>
                        </td>
                        <td style={cellStyle}>
                          {requirement.approved_test_count} / {requirement.draft_test_count} / {requirement.rejected_test_count}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </section>
        )}
      </div>
    </main>
  );
}
