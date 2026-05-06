"use client";

import type { FormEvent } from "react";
import { useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import axios from "axios";
import { Manrope } from "next/font/google";
import { toast } from "sonner";
import { register, startOAuthLogin, type OAuthProvider } from "@/api/userProvider";

const manrope = Manrope({
  subsets: ["latin"],
  variable: "--font-manrope",
});

export default function RegisterPage() {
  const router = useRouter();
  const [firstName, setFirstName] = useState("");
  const [lastName, setLastName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [agree, setAgree] = useState(false);
  const [loading, setLoading] = useState(false);
  const [oauthProvider, setOauthProvider] = useState<OAuthProvider | null>(null);
  const [error, setError] = useState<string | null>(null);

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

  const checksPassedCount = Object.values(passwordChecks).filter(Boolean).length;
  const strengthPercent = Math.round((checksPassedCount / 5) * 100);

  const missingRequirements = [
    !passwordChecks.minLength && "At least 8 characters",
    !passwordChecks.hasUpper && "One uppercase letter",
    !passwordChecks.hasLower && "One lowercase letter",
    !passwordChecks.hasNumber && "One number",
    !passwordChecks.hasSpecial && "One special character",
  ].filter(Boolean) as string[];

  const getFriendlyError = (err: unknown, fallback: string): string => {
    if (axios.isAxiosError(err)) {
      const status = err.response?.status;
      const detail = err.response?.data?.detail;
      const raw = typeof detail === "string" ? detail : err.message || "";
      const message = raw.trim();
      const lower = message.toLowerCase();

      if (status === 409 || lower.includes("already exists")) {
        return "This email is already registered. Please log in instead.";
      }
      if (status === 429 || lower.includes("rate-limit")) {
        return "Too many requests. Please wait a bit and try again.";
      }
      if (lower.includes("network") || !status) {
        return "Network issue. Please check your connection and try again.";
      }
      if (status && status >= 500) {
        return "Server issue during registration. Please try again in a moment.";
      }
      return message || fallback;
    }

    if (err instanceof Error && err.message?.trim()) {
      return err.message.trim();
    }
    return fallback;
  };

  const handleRegister = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (loading) return;
    if (!firstName || !lastName || !email || !password) {
      setError("Please fill all required fields.");
      return;
    }
    if (!agree) {
      setError("You must agree to the terms and privacy policy.");
      return;
    }
    if (missingRequirements.length > 0) {
      setError(`Password requirements missing: ${missingRequirements.join(", ")}`);
      return;
    }
    setLoading(true);
    setError(null);
    try {
      await register({
        email,
        password,
        first_name: firstName,
        last_name: lastName,
        agree_terms: agree,
      });
      toast.success("Registration successful. Please verify your email before logging in.");
      setTimeout(() => {
        router.push("/login");
      }, 1200);
    } catch (err) {
      setError(
        getFriendlyError(
          err,
          "Unexpected error during registration. Please try again."
        )
      );
    } finally {
      setLoading(false);
    }
  };

  const handleOAuthLogin = async (provider: OAuthProvider) => {
    if (loading || oauthProvider) return;

    setOauthProvider(provider);
    setError(null);

    try {
      const redirectTo = `${window.location.origin}/auth/callback`;
      const response = await startOAuthLogin(provider, redirectTo);
      window.location.href = response.data.url;
    } catch (err) {
      setError(
        getFriendlyError(
          err,
          "Unexpected error while starting social login. Please try again."
        )
      );
      setOauthProvider(null);
    }
  };

  return (
    <div className={`${manrope.variable} font-sans flex min-h-screen w-full bg-background text-foreground`}>
      {/* Top-left logo + title */}
      <div className="flex items-center absolute top-2 left-1 z-20">
        <img src="/assets/learningos-logo.png" alt="Logo" className="w-10 h-10 rounded-full"/>
        <h1 className="ml-2 text-xl font-bold text-foreground">LearningOS</h1>
      </div>

      {/* Left Panel */}
      <div className="relative hidden w-1/2 flex-col justify-center border-r border-border bg-[radial-gradient(circle_at_top_right,rgba(37,99,235,0.16),transparent_34%),linear-gradient(135deg,var(--background),color-mix(in_oklab,var(--card)_88%,#dbeafe_12%))] p-16 lg:flex">
        <h1 className="mb-6 text-5xl font-black leading-tight text-foreground">
          Unlock Your Potential With AI
        </h1>
        <p className="relative top-3 z-10 mb-10 text-lg text-muted-foreground">
          Join us to master new skills with personalized paths and get started for free today.
        </p>
        <img
          src="/assets/ai-illustration.png"
          className="w-[500px] absolute bottom-0 left-69"
          alt="AI"
        />
      </div>

      {/* Right Panel */}
      <div className="relative flex w-full flex-col items-center justify-center bg-[radial-gradient(circle_at_top_left,rgba(37,99,235,0.12),transparent_28%),linear-gradient(160deg,var(--background),color-mix(in_oklab,var(--muted)_70%,transparent))] px-6 py-20 lg:w-1/2">
        <div className="w-full max-w-[420px] rounded-[28px] border border-border bg-card/90 p-8 shadow-[0_24px_80px_rgba(15,23,42,0.12)] backdrop-blur">

          {/* Already have an account */}
          <div className="absolute top-4 right-4 text-right z-20">
            <span className="text-sm text-muted-foreground">Already have an account? </span>
            <Link href="/login" className="ml-1 text-sm font-bold text-blue-600">
              Log in
            </Link>
          </div>

          {/* Create Account Header */}
          <div className="flex items-center gap-3 mb-2 mt-12">
            <h2 className="text-4xl font-bold text-foreground">Create your account</h2>
          </div>
          <p className="mb-8 text-muted-foreground">
            Start your personalized learning journey today.
          </p>

          {/* Social Login */}
          <div className="mb-6">
            <button
              type="button"
              className="flex h-12 w-full items-center justify-center gap-2 rounded-lg border border-border bg-background text-foreground disabled:cursor-not-allowed disabled:opacity-60"
              onClick={() => handleOAuthLogin("google")}
              disabled={loading || oauthProvider !== null}
            >
              <img src="/assets/google-logo.png" alt="Google Logo" className="w-5 h-5"/>
              {oauthProvider === "google" ? "Connecting..." : "Google"}
            </button>
          </div>

          <div className="flex items-center">
  {/* Left Line */}
  <div className="h-px flex-grow bg-border"></div>

  {/* YOUR TEXT GOES HERE */}
  <span className="flex-shrink px-4 text-xs font-medium uppercase tracking-widest text-muted-foreground">
    OR REGISTER WITH EMAIL
  </span>

  {/* Right Line */}
  <div className="h-px flex-grow bg-border"></div>
</div>

          {/* Form */}
          <form className="space-y-4 mt-2" onSubmit={handleRegister}>

            {/* Name Fields */}
            <div className="flex gap-4">
              <div className="flex flex-col flex-1">
                <label className="mb-1 text-sm font-semibold text-foreground">First Name</label>
                <input
                  className="h-12 w-full rounded-lg border border-input bg-background px-4 text-foreground outline-none transition focus:border-blue-500 focus:ring-2 focus:ring-blue-100 dark:focus:ring-blue-950"
                  placeholder="Jane"
                  value={firstName}
                  onChange={(e) => setFirstName(e.target.value)}
                  required
                />
              </div>
              <div className="flex flex-col flex-1">
                <label className="mb-1 text-sm font-semibold text-foreground">Last Name</label>
                <input
                  className="h-12 w-full rounded-lg border border-input bg-background px-4 text-foreground outline-none transition focus:border-blue-500 focus:ring-2 focus:ring-blue-100 dark:focus:ring-blue-950"
                  placeholder="Doe"
                  value={lastName}
                  onChange={(e) => setLastName(e.target.value)}
                  required
                />
              </div>
            </div>

            {/* Email Field */}
            <div className="relative">
              <label className="mb-1 text-sm font-semibold text-foreground">Work Email</label>
              <input
                type="email"
                placeholder="jane@company.com"
                className="h-12 w-full rounded-xl border border-input bg-background pl-11 pr-4 text-foreground outline-none transition-all focus:border-blue-500 focus:ring-2 focus:ring-blue-100 dark:focus:ring-blue-950"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
              />
              
              <img
                src="/assets/email-icon.png"
                alt="Email Icon"
                className="absolute left-3 top-1/2 translate-y-1/7 w-5 h-5 "
              />
            </div>

            {/* Password Field */}
            <div className="relative">
              <label className="mb-1 text-sm font-semibold text-foreground">Password</label>
              <input
                type="password"
                placeholder="Min. 8 characters"
                className="h-12 w-full rounded-xl border border-input bg-background pl-11 pr-4 text-foreground outline-none transition-all focus:border-blue-500 focus:ring-2 focus:ring-blue-100 dark:focus:ring-blue-950"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
              />
              
              
              <img
                src="/assets/lock-icon.png"
                alt="Lock Icon"
                className="absolute left-3 top-1/2 translate-y-1/14 w-5 h-5"
              />
            </div>

            {/* Password Strength */}
            <div className="mt-1">
              <div className="mb-2 flex gap-1">
                {[0, 1, 2, 3, 4].map((idx) => (
                  <div
                    key={idx}
                    className={`h-1 flex-1 rounded-full ${
                      idx < checksPassedCount ? "bg-emerald-500" : "bg-border"
                    }`}
                  />
                ))}
              </div>
              <p className="text-xs text-muted-foreground">{strengthPercent}% strength</p>
              {password.length > 0 && missingRequirements.length > 0 && (
                <p className="mt-1 text-xs text-amber-600">
                  Missing: {missingRequirements.join(", ")}
                </p>
              )}
            </div>

            {/* Terms Checkbox */}
            <div className="flex items-center gap-2 mt-2">
              <input
                type="checkbox"
                id="terms"
                className="h-4 w-4 rounded border-border text-blue-600"
                checked={agree}
                onChange={(e) => setAgree(e.target.checked)}
                required
              />
              <label htmlFor="terms" className="text-sm text-muted-foreground">
                I agree to the <a href="#" className="text-blue-600">Terms</a> and <a href="#" className="text-blue-600">Privacy Policy</a>.
              </label>
            </div>

            <button
              className="flex h-12 w-full items-center justify-center gap-2 rounded-lg bg-blue-600 font-semibold text-white transition hover:bg-blue-700"
              type="submit"
              disabled={loading}
            >
              {loading ? "Creating..." : "Create Account"}
            </button>
          </form>
          {error && <p className="text-sm text-red-500 mt-3">{error}</p>}
        </div>
      </div>
    </div>
  );
}
