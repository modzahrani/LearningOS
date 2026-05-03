"use client";

import { Check, CheckCircle2, Hourglass } from "lucide-react";

import { Card, CardContent } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";

type DashboardStat = {
  title: string;
  value: string;
};

type CurrentLesson = {
  title: string;
  description: string;
  progress: number;
  completedLabel: string;
  modulesLabel: string;
  status: string;
  image: string;
};

type DailyGoal = {
  progress: number;
  message: string;
};

type Recommendation = {
  title: string;
  subtitle: string;
  duration: string;
  level: string;
  image: string;
};

type ActivityItem = {
  type: "quiz_completed" | "module_started" | "module_completed";
  title: string;
  subtitle: string;
};

type DashboardData = {
  user: {
    name: string;
    learnedMinutesToday: number;
  };
  stats: DashboardStat[];
  currentLesson: CurrentLesson;
  dailyGoal: DailyGoal;
  recommendations: Recommendation[];
  recentActivity: ActivityItem[];
};

/**
 * مؤقتًا البيانات جاية من mock object.
 * لاحقًا بدّل محتوى هذي الفنكشن بقراءة من API أو database.
 */
function getDashboardData(): DashboardData {
  return {
    user: {
      name: "Mohammed",
      learnedMinutesToday: 32,
    },
    stats: [
      { title: "Lessons complete", value: "3/14" },
      { title: "Average quiz score", value: "82%" },
      { title: "Last quiz result", value: "4/5" },
      { title: "Current level", value: "Advanced" },
    ],
    currentLesson: {
      title: "Advanced Machine Learning",
      description:
        "Master the art of Machine Learning with Python. Currently on Module 3.",
      progress: 85,
      completedLabel: "85% Completed",
      modulesLabel: "12/14 Modules",
      status: "IN PROGRESS",
      image:
        "https://images.unsplash.com/photo-1555949963-aa79dcee981c?q=80&w=1200&auto=format&fit=crop",
    },
    dailyGoal: {
      progress: 30,
      message: "Great work! Keep Going.",
    },
    recommendations: [
      {
        title: "Deep Learning",
        subtitle: "Based on your interest in Machine Learning",
        duration: "1 hour 51 min",
        level: "Advanced",
        image:
          "https://images.unsplash.com/photo-1515879218367-8466d910aaa4?q=80&w=1200&auto=format&fit=crop",
      },
      {
        title: "Model Deployment",
        subtitle: "Based on your interest in Machine Learning",
        duration: "60 min",
        level: "Advanced",
        image:
          "https://images.unsplash.com/photo-1555949963-aa79dcee981c?q=80&w=1200&auto=format&fit=crop",
      },
    ],
    recentActivity: [
      {
        type: "quiz_completed",
        title: "Completed Quiz",
        subtitle: "“Advanced Machine Learning Quiz 2”",
      },
      {
        type: "module_started",
        title: "Started Module 3",
        subtitle: "“Advanced Machine Learning Module 3”",
      },
      {
        type: "module_completed",
        title: "Completed Module 2",
        subtitle: "“Advanced Machine Learning Module 2”",
      },
    ],
  };
}

export default function DashboardPage() {
  const dashboardData = getDashboardData();

  return (
    <main className="min-h-screen bg-[#f4f7fb] px-6 py-8 md:px-8">
      <div className="mx-auto max-w-7xl space-y-6">
        <div>
          <h1 className="text-4xl font-bold tracking-tight text-slate-900">
            Welcome back, {dashboardData.user.name}! 👋
          </h1>
          <p className="mt-2 text-base text-slate-400">
            You&apos;ve learned for {dashboardData.user.learnedMinutesToday} minutes
            today, keep it up!
          </p>
        </div>

        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
          {dashboardData.stats.map((stat) => (
            <StatCard key={stat.title} title={stat.title} value={stat.value} />
          ))}
        </div>

        <div className="grid gap-5 xl:grid-cols-[2fr_1fr]">
          <Card className="rounded-2xl border-slate-200 bg-white shadow-sm">
            <CardContent className="p-6">
              <div className="space-y-5">
                <div className="flex items-start justify-between gap-4">
                  <div>
                    <h2 className="text-2xl font-bold text-slate-900">
                      Current Active Lesson
                    </h2>
                    <p className="mt-1 text-sm text-slate-400">
                      Continue where you left off
                    </p>
                  </div>

                  <Badge className="rounded-full bg-blue-100 text-blue-700 hover:bg-blue-100">
                    {dashboardData.currentLesson.status}
                  </Badge>
                </div>

                <div className="grid gap-5 md:grid-cols-[120px_1fr] md:items-center">
                  <div className="overflow-hidden rounded-2xl bg-slate-200">
                    <img
                      src={dashboardData.currentLesson.image}
                      alt={dashboardData.currentLesson.title}
                      className="h-[120px] w-full object-cover"
                    />
                  </div>

                  <div className="space-y-4">
                    <div>
                      <h3 className="text-3xl font-bold text-slate-900">
                        {dashboardData.currentLesson.title}
                      </h3>
                      <p className="mt-1 text-base text-slate-400">
                        {dashboardData.currentLesson.description}
                      </p>
                    </div>

                    <div className="flex items-center justify-between text-sm font-medium text-slate-700">
                      <span>{dashboardData.currentLesson.completedLabel}</span>
                      <span>{dashboardData.currentLesson.modulesLabel}</span>
                    </div>

                    <Progress
                      value={dashboardData.currentLesson.progress}
                      className="h-3"
                    />

                    <Button className="rounded-xl bg-blue-600 px-6 text-white hover:bg-blue-700">
                      Continue Learning →
                    </Button>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="rounded-2xl border-slate-200 bg-white shadow-sm">
            <CardContent className="flex h-full flex-col items-center justify-center p-6 text-center">
              <h2 className="text-2xl font-bold text-slate-900">Daily Goal</h2>

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
                    strokeDashoffset={289 - (289 * dashboardData.dailyGoal.progress) / 100}
                  />
                </svg>

                <div className="absolute inset-0 flex items-center justify-center text-center">
                  <div>
                    <p className="text-2xl font-bold text-slate-900">
                      {dashboardData.dailyGoal.progress}%
                    </p>
                    <p className="text-sm text-slate-500">Complete</p>
                  </div>
                </div>
              </div>

              <p className="text-sm text-slate-400">
                {dashboardData.dailyGoal.message}
              </p>
            </CardContent>
          </Card>

          <Card className="rounded-2xl border-slate-200 bg-white shadow-sm">
            <CardContent className="p-6">
              <div className="space-y-5">
                <h2 className="text-2xl font-bold text-slate-900">
                  ✨ AI Recommended for you
                </h2>

                <div className="grid gap-5 md:grid-cols-2">
                  {dashboardData.recommendations.map((item) => (
                    <RecommendationCard
                      key={item.title}
                      title={item.title}
                      subtitle={item.subtitle}
                      duration={item.duration}
                      level={item.level}
                      image={item.image}
                    />
                  ))}
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="rounded-2xl border-slate-200 bg-white shadow-sm">
            <CardContent className="p-6">
              <div className="space-y-5">
                <h2 className="text-2xl font-bold text-slate-900">
                  Recent Activity
                </h2>

                <div className="space-y-4 text-sm text-slate-600">
                  {dashboardData.recentActivity.map((activity, index) => (
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
    <Card className="rounded-2xl border-slate-200 bg-white shadow-sm">
      <CardContent className="p-5">
        <p className="text-sm text-slate-400">{title}</p>
        <h3 className="mt-3 text-4xl font-bold text-slate-900">{value}</h3>
      </CardContent>
    </Card>
  );
}

function RecommendationCard({
  title,
  subtitle,
  duration,
  level,
  image,
}: {
  title: string;
  subtitle: string;
  duration: string;
  level: string;
  image: string;
}) {
  return (
    <Card className="overflow-hidden rounded-2xl border-slate-200 bg-white shadow-sm">
      <div className="h-44 overflow-hidden bg-slate-200">
        <img src={image} alt={title} className="h-full w-full object-cover" />
      </div>

      <CardContent className="space-y-3 p-4">
        <div>
          <h3 className="text-2xl font-bold text-slate-900">{title}</h3>
          <p className="mt-1 text-sm text-slate-400">{subtitle}</p>
        </div>

        <div className="flex items-center justify-between text-sm text-slate-400">
          <span>{duration}</span>
          <span>{level}</span>
        </div>
      </CardContent>
    </Card>
  );
}

function ActivityRow({
  type,
  title,
  subtitle,
}: {
  type: ActivityItem["type"];
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
        <p className="font-medium">{title}</p>
        <p className="text-slate-400">{subtitle}</p>
      </div>
    </div>
  );
}