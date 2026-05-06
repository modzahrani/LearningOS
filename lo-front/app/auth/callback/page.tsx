"use client";

import { useEffect, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import axios from "axios";

import { getAssignedLessons } from "@/api/lessonsProvider";
import { getSelectedPath } from "@/api/pathProvider";
import { createOAuthSession } from "@/api/userProvider";

const parseHashParams = (hash: string) => {
  const value = hash.startsWith("#") ? hash.slice(1) : hash;
  return new URLSearchParams(value);
};

export default function AuthCallbackPage() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let isMounted = true;

    const finishLogin = async () => {
      const hashParams = parseHashParams(window.location.hash);
      const accessToken =
        hashParams.get("access_token") || searchParams.get("access_token");
      const errorDescription =
        searchParams.get("error_description") ||
        hashParams.get("error_description") ||
        searchParams.get("error");

      if (errorDescription) {
        if (isMounted) {
          setError(decodeURIComponent(errorDescription));
        }
        return;
      }

      if (!accessToken) {
        if (isMounted) {
          setError("Social login did not return an access token.");
        }
        return;
      }

      try {
        await createOAuthSession(accessToken);
        window.history.replaceState({}, document.title, window.location.pathname);

        try {
          await getSelectedPath();
        } catch (err) {
          if (axios.isAxiosError(err) && err.response?.status === 404) {
            router.replace("/path-select");
            return;
          }
          throw err;
        }

        try {
          await getAssignedLessons();
          router.replace("/dashboard");
          return;
        } catch (err) {
          if (axios.isAxiosError(err) && err.response?.status === 404) {
            router.replace("/questionnaire");
            return;
          }
          throw err;
        }
      } catch (err) {
        const message =
          axios.isAxiosError(err)
            ? typeof err.response?.data?.detail === "string"
              ? err.response.data.detail
              : err.message
            : err instanceof Error
              ? err.message
              : "Social login failed. Please try again.";

        if (isMounted) {
          setError(message);
        }
      }
    };

    finishLogin();

    return () => {
      isMounted = false;
    };
  }, [router, searchParams]);

  return (
    <main className="flex min-h-screen items-center justify-center bg-[#f4f7fb] px-6">
      <div className="w-full max-w-md rounded-2xl border border-slate-200 bg-white p-8 text-center shadow-sm">
        <h1 className="text-2xl font-bold text-slate-900">
          {error ? "Login failed" : "Finishing sign in"}
        </h1>
        <p className="mt-3 text-sm text-slate-500">
          {error
            ? error
            : "We’re connecting your account and getting your workspace ready."}
        </p>
        {error && (
          <button
            type="button"
            className="mt-6 rounded-xl bg-blue-600 px-5 py-3 text-sm font-semibold text-white hover:bg-blue-700"
            onClick={() => router.replace("/login")}
          >
            Back to login
          </button>
        )}
      </div>
    </main>
  );
}
