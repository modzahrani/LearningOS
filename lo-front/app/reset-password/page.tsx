"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import axios from "axios";

import { resetPassword } from "@/api/userProvider";

const parseHashParams = (hash: string) => {
  const value = hash.startsWith("#") ? hash.slice(1) : hash;
  return new URLSearchParams(value);
};

export default function ResetPasswordPage() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const [accessToken, setAccessToken] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  useEffect(() => {
    const hashParams = parseHashParams(window.location.hash);
    const token = hashParams.get("access_token") || searchParams.get("access_token") || "";
    const errorDescription =
      searchParams.get("error_description") ||
      hashParams.get("error_description") ||
      searchParams.get("error");

    if (errorDescription) {
      setError(decodeURIComponent(errorDescription));
      return;
    }

    if (!token) {
      setError("This password reset link is missing the recovery token.");
      return;
    }

    setAccessToken(token);
    window.history.replaceState({}, document.title, window.location.pathname);
  }, [searchParams]);

  const passwordChecks = useMemo(
    () => ({
      minLength: password.length >= 8,
      hasUpper: /[A-Z]/.test(password),
      hasLower: /[a-z]/.test(password),
      hasNumber: /\d/.test(password),
      hasSpecial: /[^A-Za-z0-9]/.test(password),
    }),
    [password]
  );

  const missingRequirements = [
    !passwordChecks.minLength && "8+ characters",
    !passwordChecks.hasUpper && "an uppercase letter",
    !passwordChecks.hasLower && "a lowercase letter",
    !passwordChecks.hasNumber && "a number",
    !passwordChecks.hasSpecial && "a special character",
  ].filter(Boolean) as string[];

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
    if (!accessToken) {
      setError("This password reset link is invalid or incomplete.");
      return;
    }
    if (!password) {
      setError("Please enter a new password.");
      return;
    }
    if (password !== confirmPassword) {
      setError("Passwords do not match.");
      return;
    }
    if (missingRequirements.length > 0) {
      setError(`Password must include ${missingRequirements.join(", ")}.`);
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await resetPassword({
        access_token: accessToken,
        new_password: password,
      });
      setSuccess(response.data.detail);
      setTimeout(() => {
        router.replace("/login");
      }, 1400);
    } catch (err) {
      setError(
        getFriendlyError(err, "We couldn't reset your password right now. Please try again.")
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="flex min-h-screen items-center justify-center bg-[radial-gradient(circle_at_top,#ecf4ff_0%,#f9fbff_36%,#e8eef9_100%)] px-6">
      <section className="w-full max-w-lg rounded-[30px] border border-white/70 bg-white/90 p-8 shadow-[0_24px_80px_rgba(46,82,140,0.14)] backdrop-blur">
        <div className="mb-8">
          <p className="text-sm font-semibold uppercase tracking-[0.24em] text-blue-600">
            Secure Reset
          </p>
          <h1 className="mt-3 text-3xl font-bold text-slate-950">Choose a new password</h1>
          <p className="mt-3 text-sm leading-6 text-slate-600">
            Use a strong password you haven&apos;t used before on this account.
          </p>
        </div>

        <div className="space-y-4">
          <div>
            <label className="mb-2 block text-sm font-semibold text-slate-800">
              New password
            </label>
            <input
              type="password"
              value={password}
              onChange={(event) => setPassword(event.target.value)}
              placeholder="Enter your new password"
              className="w-full rounded-2xl border border-slate-200 bg-white px-4 py-3 text-sm outline-none transition focus:border-blue-500 focus:ring-2 focus:ring-blue-100"
            />
          </div>

          <div>
            <label className="mb-2 block text-sm font-semibold text-slate-800">
              Confirm password
            </label>
            <input
              type="password"
              value={confirmPassword}
              onChange={(event) => setConfirmPassword(event.target.value)}
              placeholder="Re-enter your new password"
              className="w-full rounded-2xl border border-slate-200 bg-white px-4 py-3 text-sm outline-none transition focus:border-blue-500 focus:ring-2 focus:ring-blue-100"
            />
          </div>
        </div>

        <p className="mt-4 text-xs leading-5 text-slate-500">
          Password must include at least 8 characters, an uppercase letter, a lowercase
          letter, a number, and a special character.
        </p>

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
          disabled={loading || !accessToken}
          className="mt-6 w-full rounded-2xl bg-blue-600 px-4 py-3 text-sm font-semibold text-white transition hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-60"
        >
          {loading ? "Updating password..." : "Update password"}
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
