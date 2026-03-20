"use client";

import { useState } from "react";
import { Check } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress"
import { cn } from "@/lib/utils";

type PathOption = {
  id: "student" | "individual" | "enterprise";
  title: string;
  description: string;
  image: string;
};

const PATHS: PathOption[] = [
  {
    id: "student",
    title: "Student",
    description:
      "For academic excellence, track grades, organize assignments, and master subjects.",
    image:
      "https://images.unsplash.com/photo-1509062522246-3755977927d7?q=80&w=1200&auto=format&fit=crop",
  },
  {
    id: "individual",
    title: "Individual",
    description:
      "For personal growth, upskill at your own pace with a personalized roadmap.",
    image:
      "https://images.unsplash.com/photo-1516321318423-f06f85e504b3?q=80&w=1200&auto=format&fit=crop",
  },
  {
    id: "enterprise",
    title: "Enterprise",
    description:
      "For teams, manage progress, assign modules, and scale workforce training effectively.",
    image:
      "https://images.unsplash.com/photo-1552664730-d307ca884978?q=80&w=1200&auto=format&fit=crop",
  },
];

export default function PathSelectPage() {
  const [selectedPath, setSelectedPath] =
    useState<PathOption["id"]>("individual");
  const [saved, setSaved] = useState(false);

  const handleContinue = () => {
    localStorage.setItem("learningos_selected_path", selectedPath);
    setSaved(true);
    console.log("Selected path:", selectedPath);
  };

  return (
    <main className="min-h-screen bg-[#f5f7fb] px-4 py-8 md:px-8">
      <div className="mx-auto max-w-6xl">
        {/* Top row */}
        <div className="mb-10 flex flex-col items-center justify-between gap-6 md:flex-row">
          <div className="w-full max-w-md">
            <div className="mb-2 flex items-center justify-between text-sm">
              <span className="font-medium text-slate-700">Step 1 of 4</span>
              <span className="text-slate-400">25% completed</span>
            </div>

            <Progress value={25} className="h-2 bg-blue-100" />
          </div>

          <div className="flex items-center gap-6 text-sm text-slate-400">
            <button className="transition hover:text-slate-600">Help</button>
            <button className="transition hover:text-slate-600">Login</button>
          </div>
        </div>
        {/* Heading */}
        <div className="mb-10 text-center">
          <h1 className="text-3xl font-bold tracking-tight text-slate-900 md:text-4xl">
            Choose your learning journal
          </h1>
          <p className="mx-auto mt-3 max-w-2xl text-sm leading-6 text-slate-500 md:text-base">
            We&apos;ll tailor the AI curriculum based on your goals. Select the
            path that best describes your current status.
          </p>
        </div>

        {/* Cards */}
        <div className="grid gap-6 md:grid-cols-3">
          {PATHS.map((path) => {
            const isSelected = selectedPath === path.id;

            return (
              <button
                key={path.id}
                type="button"
                onClick={() => {
                  setSelectedPath(path.id);
                  setSaved(false);
                }}
                className={cn(
                  "relative rounded-2x1 border bg-white p-4 text-left shadow-sm transition-all hover:shadow-md",
                  isSelected
                    ? "border-blue-600 ring-2 ring-blue-200"
                    : "border-slate-200",
                )}
              >
                {isSelected && (
                  <div className="absolute -right-2 -top-2 flex h-8 w-8 items-center justify-center rounded-full bg-blue-600 text-white shadow-lg">
                    <Check className="h-4 w-4" />
                  </div>
                )}

                <div className="overflow-hidden rounded-xl">
                  <img
                    src={path.image}
                    alt={path.title}
                    className="h-48 w-full object-cover"
                  />
                </div>
                {/* radio circle */}
                <div className="mt-4 flex items-start justify-between gap-4">
                  <div>
                    <h2 className="text-xl font-semibold text-slate-900">
                      {path.title}
                    </h2>
                    <p className="mt-2 text-sm leading-6 text-slate-500">
                      {path.description}
                    </p>
                  </div>

                  <div
                    className={cn(
                      "mt-1 flex size-6 shrink-0 items-center justify-center rounded-full border-2",
                      isSelected ? "border-blue-600" : "border-slate-300",
                    )}
                  >
                    {isSelected && (
                      <div className="size-3 rounded-full bg-blue-600" />
                    )}
                  </div>
                </div>
              </button>
            );
          })}
        </div>

        {/* Continue button */}
        <div className="mt-14 flex flex-col items-center justify-center gap-4">
          <Button
            onClick={handleContinue}
            className="h-12 rounded-xl bg-blue-600 px-10 text-base font-semibold text-white hover:bg-blue-700"
          >
            Continue →
          </Button>

        
        </div>
      </div>
    </main>
  );
}