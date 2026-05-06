"use client";

import { useState } from "react";
import Link from "next/link";
import axios from "axios";

import { forgotPassword } from "@/api/userProvider";

export default function ForgotPasswordPage() {
  const [email, setEmail] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  const getFriendlyError = (err: unknown, fallback: string): string => {
    if (axios.isAxiosError(err)) {
      const detail = err.response?.data?.detail;
      if (typeof detail === "string" && detail.trim()) {
        return detail.trim();
      }
      return err.message || fallback;
    }

    if (err instanceof Error && err.message.trim()) {
      return err.message.trim();
    }

    return fallback;
  };

  const handleSubmit = async () => {
    if (loading) return;
    if (!email.trim()) {
      setError("Please enter your email address.");
      return;
    }

    setLoading(true);
    setError(null);
    setSuccess(null);

    try {
      const redirectTo = `${window.location.origin}/reset-password`;
      const response = await forgotPassword({
        email: email.trim(),
        redirect_to: redirectTo,
      });
      setSuccess(response.data.detail);
    } catch (err) {
      setError(
        getFriendlyError(err, "We couldn't send a reset email right now. Please try again.")
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="flex min-h-screen items-center justify-center bg-[linear-gradient(160deg,#f8fbff_0%,#e9f0fb_52%,#dfe8f8_100%)] px-6">
      <section className="w-full max-w-md rounded-[28px] border border-white/70 bg-white/90 p-8 shadow-[0_24px_80px_rgba(46,82,140,0.14)] backdrop-blur">
        <div className="mb-8">
          <p className="text-sm font-semibold uppercase tracking-[0.24em] text-blue-600">
            Account Recovery
          </p>
          <h1 className="mt-3 text-3xl font-bold text-slate-950">Forgot your password?</h1>
          <p className="mt-3 text-sm leading-6 text-slate-600">
            Enter the email tied to your account and we&apos;ll send you a reset link.
          </p>
        </div>

        <label className="mb-2 block text-sm font-semibold text-slate-800">
          Email address
        </label>
        <input
          type="email"
          value={email}
          onChange={(event) => setEmail(event.target.value)}
          placeholder="you@example.com"
          className="w-full rounded-2xl border border-slate-200 bg-white px-4 py-3 text-sm outline-none transition focus:border-blue-500 focus:ring-2 focus:ring-blue-100"
        />

        {error && (
          <p className="mt-4 rounded-2xl border border-red-100 bg-red-50 px-4 py-3 text-sm text-red-700">
            {error}
          </p>
        )}

        {success && (
          <p className="mt-4 rounded-2xl border border-emerald-100 bg-emerald-50 px-4 py-3 text-sm text-emerald-700">
            {success}
          </p>
        )}

        <button
          type="button"
          onClick={handleSubmit}
          disabled={loading}
          className="mt-6 w-full rounded-2xl bg-blue-600 px-4 py-3 text-sm font-semibold text-white transition hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-60"
        >
          {loading ? "Sending reset link..." : "Send reset link"}
        </button>

        <div className="mt-6 text-center text-sm text-slate-600">
          <Link href="/login" className="font-semibold text-blue-600 hover:text-blue-700">
            Back to login
          </Link>
        </div>
      </section>
    </main>
  );
}
