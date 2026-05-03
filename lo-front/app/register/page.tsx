"use client";

import type { FormEvent } from "react";
import { useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import axios from "axios";
import { Manrope } from "next/font/google";
import { register } from "@/api/userProvider";

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
  const [error, setError] = useState<string | null>(null);
  const [successToast, setSuccessToast] = useState<string | null>(null);

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
      setSuccessToast("Registration successful. Confirm your email.");
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

  return (
    <div className={`${manrope.variable} font-sans flex w-full h-screen`}>
      {successToast && (
        <div className="fixed right-4 top-4 z-50 rounded-lg bg-emerald-600 px-4 py-3 text-sm font-semibold text-white shadow-lg">
          {successToast}
        </div>
      )}
      
      {/* Top-left logo + title */}
      <div className="flex items-center absolute top-2 left-1 z-20">
        <img src="/assets/learningos-logo.png" alt="Logo" className="w-10 h-10 rounded-full"/>
        <h1 className="text-xl font-bold ml-2">LearningOS</h1>
      </div>

      {/* Left Panel */}
      <div className="hidden lg:flex w-1/2 bg-white border-r p-16 flex-col justify-center relative">
        <h1 className="text-5xl font-black leading-tight mb-6">
          Unlock Your Potential With AI
        </h1>
        <p className="top-3 text-[#4C669A] text-lg mb-10 relative z-10">
          Join us to master new skills with personalized paths and get started for free today.
        </p>
        <img
          src="/assets/ai-illustration.png"
          className="w-[500px] absolute bottom-0 left-69"
          alt="AI"
        />
      </div>

      {/* Right Panel */}
      <div className="w-full lg:w-1/2 flex flex-col items-center justify-center bg-gray-100 relative">
        <div className="w-[420px]">

          {/* Already have an account */}
          <div className="absolute top-4 right-4 text-right z-20">
            <span className="text-gray-500 text-sm">Already have an account? </span>
            <Link href="/login" className="text-blue-600 font-bold text-sm ml-1">
              Log in
            </Link>
          </div>

          {/* Create Account Header */}
          <div className="flex items-center gap-3 mb-2 mt-12">
            <h2 className="text-4xl font-bold">Create your account</h2>
          </div>
          <p className="text-gray-500 mb-8">
            Start your personalized learning journey today.
          </p>

          {/* Social Login */}
          <div className="grid grid-cols-2 gap-4 mb-6">
            <button type="button" className="border h-12 rounded-lg flex items-center justify-center gap-2 bg-white">
              <img src="/assets/google-logo.png" alt="Google Logo" className="w-5 h-5"/>
              Google
            </button>
            <button type="button" className="border h-12 rounded-lg flex items-center justify-center gap-2 bg-white">
              <img src="/assets/microsoft-logo.png" alt="Microsoft Logo" className="w-5 h-5"/>
              Microsoft
            </button>
          </div>

          <div className="flex items-center">
  {/* Left Line */}
  <div className="flex-grow h-px bg-gray-200"></div>

  {/* YOUR TEXT GOES HERE */}
  <span className="flex-shrink px-4 text-xs font-medium text-gray-400 uppercase tracking-widest">
    OR REGISTER WITH EMAIL
  </span>

  {/* Right Line */}
  <div className="flex-grow h-px bg-gray-200"></div>
</div>

          {/* Form */}
          <form className="space-y-4 mt-2" onSubmit={handleRegister}>

            {/* Name Fields */}
            <div className="flex gap-4">
              <div className="flex flex-col flex-1">
                <label className="text-sm font-semibold mb-1">First Name</label>
                <input
                  className="border rounded-lg h-12 px-4 w-full"
                  placeholder="Jane"
                  value={firstName}
                  onChange={(e) => setFirstName(e.target.value)}
                  required
                />
              </div>
              <div className="flex flex-col flex-1">
                <label className="text-sm font-semibold mb-1">Last Name</label>
                <input
                  className="border rounded-lg h-12 px-4 w-full"
                  placeholder="Doe"
                  value={lastName}
                  onChange={(e) => setLastName(e.target.value)}
                  required
                />
              </div>
            </div>

            {/* Email Field */}
            <div className="relative">
              <label className="text-sm font-semibold mb-1">Work Email</label>
              <input
                type="email"
                placeholder="jane@company.com"
                className="w-full h-12 pl-11 pr-4 rounded-xl  border border-slate-200 focus:border-blue-500 focus:ring-1 focus:ring-blue-500 outline-none transition-all "
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
              <label className="text-sm font-semibold mb-1">Password</label>
              <input
                type="password"
                placeholder="Min. 8 characters"
                className="w-full h-12 pl-11 pr-4 rounded-xl  border border-slate-200  focus:border-blue-500 focus:ring-1 focus:ring-blue-500 outline-none transition-all "
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
                      idx < checksPassedCount ? "bg-emerald-500" : "bg-gray-200"
                    }`}
                  />
                ))}
              </div>
              <p className="text-xs text-gray-500">{strengthPercent}% strength</p>
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
                className="w-4 h-4 text-blue-600 border-gray-300 rounded"
                checked={agree}
                onChange={(e) => setAgree(e.target.checked)}
                required
              />
              <label htmlFor="terms" className="text-sm text-gray-500">
                I agree to the <a href="#" className="text-blue-600">Terms</a> and <a href="#" className="text-blue-600">Privacy Policy</a>.
              </label>
            </div>

            <button
              className="w-full h-12 bg-blue-600 text-white rounded-lg font-semibold flex items-center justify-center gap-2"
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
