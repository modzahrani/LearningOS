"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { Check, CheckCircle2, Hourglass } from "lucide-react";

import { getDashboard } from "@/api/dashboardProvider";
import type {
  DashboardActivityItem,
  DashboardResponse,
} from "@/api/dashboardProvider";
import { Card, CardContent } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";

const getErrorMessage = (err: unknown, fallback: string) => {
  if (err && typeof err === "object" && "response" in err) {
    return (err as { response?: { data?: { detail?: string } } }).response?.data?.detail || fallback;
  }
  if (err instanceof Error && err.message.trim()) {
    return err.message;
  }
  return fallback;
};

const CURRENT_LESSON_IMAGE =
  "https://images.unsplash.com/photo-1555949963-aa79dcee981c?q=80&w=1200&auto=format&fit=crop";

const getStatValue = (stats: DashboardResponse["stats"], title: string) => {
  return stats.find((stat) => stat.title.toLowerCase() === title.toLowerCase())?.value || "";
};

const getRoleLabel = (description: string) => {
  const match = description.match(/continue your\s+(.+?)\s+path/i);
  if (!match?.[1]) return "Learning";
  return match[1].charAt(0).toUpperCase() + match[1].slice(1);
};

export default function DashboardPage() {
  const router = useRouter();
  const [dashboardData, setDashboardData] = useState<DashboardResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const loadDashboard = async () => {
      setLoading(true);
      setError(null);

      try {
        const response = await getDashboard();
        setDashboardData(response.data);
      } catch (err: unknown) {
        setError(getErrorMessage(err, "Failed to load dashboard"));
      } finally {
        setLoading(false);
      }
    };

    loadDashboard();
  }, []);

  if (loading) {
    return (
      <main className="min-h-screen bg-background px-6 py-8 md:px-8">
        <div className="mx-auto max-w-7xl">
          <Card className="rounded-2xl border-slate-200 bg-white shadow-sm dark:border-slate-800 dark:bg-black">
            <CardContent className="p-6 text-slate-500 dark:text-slate-400">
              Loading your dashboard...
            </CardContent>
          </Card>
        </div>
      </main>
    );
  }

  if (error || !dashboardData) {
    return (
      <main className="min-h-screen bg-background px-6 py-8 md:px-8">
        <div className="mx-auto max-w-7xl">
          <Card className="rounded-2xl border-red-200 bg-white shadow-sm dark:border-red-900 dark:bg-black">
            <CardContent className="p-6">
              <p className="text-sm text-red-600">
                {error || "Failed to load dashboard"}
              </p>
            </CardContent>
          </Card>
        </div>
      </main>
    );
  }

  const currentLevelValue = getStatValue(dashboardData.stats, "Current level");
  const rolePathLabel = getRoleLabel(dashboardData.current_lesson.description);

  return (
    <main className="min-h-screen bg-background px-6 py-8 md:px-8">
      <div className="mx-auto max-w-7xl space-y-6">
        <div>
          <h1 className="text-4xl font-bold tracking-tight text-slate-900 dark:text-slate-100">
            Welcome back, {dashboardData.user.name}! 👋
          </h1>
          <p className="mt-2 text-base text-slate-400 dark:text-slate-500">
            You&apos;ve learned for {dashboardData.user.learned_minutes_today} minutes
            today, keep it up!
          </p>
        </div>

        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
          {dashboardData.stats.map((stat) => (
            <StatCard key={stat.title} title={stat.title} value={stat.value} />
          ))}
        </div>

        <div className="grid gap-5 xl:grid-cols-[2fr_1fr]">
          <Card className="rounded-2xl border-slate-200 bg-white shadow-sm dark:border-slate-800 dark:bg-black">
            <CardContent className="p-6">
              <div className="space-y-5">
                <div className="flex items-start justify-between gap-4">
                  <div>
                    <h2 className="text-2xl font-bold text-slate-900 dark:text-slate-100">
                      Current Active Lesson
                    </h2>
                    <p className="mt-1 text-sm text-slate-400 dark:text-slate-500">
                      Continue where you left off
                    </p>
                  </div>

                  <Badge className="rounded-full bg-blue-100 text-blue-700 hover:bg-blue-100">
                    {dashboardData.current_lesson.status}
                  </Badge>
                </div>

                <div className="grid gap-5 md:grid-cols-[120px_1fr] md:items-center">
                  <div className="overflow-hidden rounded-2xl bg-slate-200 dark:bg-slate-950">
                    <img
                      src={CURRENT_LESSON_IMAGE}
                      alt={dashboardData.current_lesson.title}
                      className="h-[120px] w-full object-cover"
                    />
                  </div>

                  <div className="space-y-4">
                    <div>
                      <h3 className="text-3xl font-bold text-slate-900 dark:text-slate-100">
                        {dashboardData.current_lesson.title}
                      </h3>
                      <p className="mt-1 text-base text-slate-400 dark:text-slate-500">
                        {dashboardData.current_lesson.description}
                      </p>
                    </div>

                    <div className="flex items-center justify-between text-sm font-medium text-slate-700 dark:text-slate-300">
                      <span>{dashboardData.current_lesson.completed_label}</span>
                    </div>

                    <Progress
                      value={dashboardData.current_lesson.progress}
                      className="h-3"
                    />

                    <Button
                      className="rounded-xl bg-blue-600 px-6 text-white hover:bg-blue-700"
                      onClick={() =>
                        dashboardData.current_lesson.source
                          ? router.push(
                              `/lesson-details?source=${encodeURIComponent(
                                dashboardData.current_lesson.source
                              )}`
                            )
                          : router.push("/lessons")
                      }
                    >
                      Continue Learning →
                    </Button>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="rounded-2xl border-slate-200 bg-white shadow-sm dark:border-slate-800 dark:bg-black">
            <CardContent className="flex h-full flex-col items-center justify-center p-6 text-center">
              <h2 className="text-2xl font-bold text-slate-900 dark:text-slate-100">Daily Goal</h2>

              <div className="relative my-6 h-36 w-36">
                <svg className="h-36 w-36 -rotate-90" viewBox="0 0 120 120">
                  <circle
                    cx="60"
                    cy="60"
                    r="46"
                    fill="none"
                    stroke="#e5e7eb"
                    strokeWidth="10"
                  />
                  <circle
                    cx="60"
                    cy="60"
                    r="46"
                    fill="none"
                    stroke="#2563eb"
                    strokeWidth="10"
                    strokeLinecap="round"
                    strokeDasharray="289"
                    strokeDashoffset={289 - (289 * dashboardData.daily_goal.progress) / 100}
                  />
                </svg>

                <div className="absolute inset-0 flex items-center justify-center text-center">
                  <div>
                    <p className="text-2xl font-bold text-slate-900 dark:text-slate-100">
                      {dashboardData.daily_goal.progress}%
                    </p>
                    <p className="text-sm text-slate-500 dark:text-slate-400">Complete</p>
                  </div>
                </div>
              </div>

              <p className="text-sm text-slate-400 dark:text-slate-500">
                {dashboardData.daily_goal.message}
              </p>
            </CardContent>
          </Card>

          <Card className="rounded-2xl border-slate-200 bg-white shadow-sm dark:border-slate-800 dark:bg-black">
            <CardContent className="p-6">
              <div className="flex items-center justify-between gap-4">
                <div>
                  <h2 className="text-2xl font-bold text-foreground">
                    Learning Snapshot
                  </h2>
                  <p className="mt-1 text-sm text-muted-foreground">
                    A quick view of your current path and momentum.
                  </p>
                </div>
                <Badge className="rounded-full bg-blue-100 text-blue-700 hover:bg-blue-100">
                  {rolePathLabel} Path
                </Badge>
              </div>

              <div className="mt-6 grid gap-4 md:grid-cols-2">
                <SnapshotBlock
                  label="Current Level"
                  value={currentLevelValue || "Beginner"}
                  helper="Adapted from your questionnaire and progress"
                />
                <SnapshotBlock
                  label="Current Progress"
                  value={`${dashboardData.current_lesson.progress}%`}
                  helper="Progress in the lesson you are actively working on"
                />
              </div>
            </CardContent>
          </Card>

          <Card className="rounded-2xl border-slate-200 bg-white shadow-sm dark:border-slate-800 dark:bg-black">
            <CardContent className="p-6">
              <div className="space-y-5">
                <h2 className="text-2xl font-bold text-slate-900 dark:text-slate-100">
                  Recent Activity
                </h2>

                <div className="space-y-4 text-sm text-slate-600 dark:text-slate-300">
                  {dashboardData.recent_activity.map((activity, index) => (
                    <ActivityRow
                      key={`${activity.title}-${index}`}
                      type={activity.type}
                      title={activity.title}
                      subtitle={activity.subtitle}
                    />
                  ))}
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </main>
  );
}

function StatCard({
  title,
  value,
}: {
  title: string;
  value: string;
}) {
  return (
    <Card className="rounded-2xl border-slate-200 bg-white shadow-sm dark:border-slate-800 dark:bg-black">
      <CardContent className="p-5">
        <p className="text-sm text-slate-400 dark:text-slate-500">{title}</p>
        <h3 className="mt-3 text-4xl font-bold text-slate-900 dark:text-slate-100">{value}</h3>
      </CardContent>
    </Card>
  );
}

function SnapshotBlock({
  label,
  value,
  helper,
}: {
  label: string;
  value: string;
  helper: string;
}) {
  return (
    <div className="rounded-2xl border border-slate-200 bg-slate-50 p-4 dark:border-slate-800 dark:bg-slate-950">
      <p className="text-sm text-slate-400">{label}</p>
      <p className="mt-2 text-2xl font-bold text-slate-900 dark:text-slate-100">{value}</p>
      <p className="mt-2 text-sm leading-6 text-slate-500">{helper}</p>
    </div>
  );
}

function ActivityRow({
  type,
  title,
  subtitle,
}: {
  type: DashboardActivityItem["type"];
  title: string;
  subtitle: string;
}) {
  const icon =
    type === "quiz_completed" ? (
      <Check className="mt-0.5 h-4 w-4 text-violet-600" />
    ) : type === "module_started" ? (
      <Hourglass className="mt-0.5 h-4 w-4 text-amber-500" />
    ) : (
      <CheckCircle2 className="mt-0.5 h-4 w-4 text-violet-600" />
    );

  return (
    <div className="flex items-start gap-3">
      {icon}
      <div>
        <p className="font-medium dark:text-slate-100">{title}</p>
        <p className="text-slate-400 dark:text-slate-500">{subtitle}</p>
      </div>
    </div>
  );
}
