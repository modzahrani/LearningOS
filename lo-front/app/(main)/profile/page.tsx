"use client";

import { useEffect, useMemo, useState } from "react";
import type { ReactNode } from "react";
import { useRouter } from "next/navigation";
import {
  ArrowRight,
  Award,
  BookCheck,
  Clock3,
  GraduationCap,
  Mail,
  Sparkles,
  Trophy,
} from "lucide-react";

import { getProfile } from "@/api/userProvider";
import type { ProfileResponse } from "@/api/userProvider";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";

const levelLabel = (level: number) => {
  if (level === 1) return "Beginner";
  if (level === 2) return "Intermediate";
  return "Advanced";
};

const difficultyLabel = (difficulty: string) => {
  if (!difficulty) return "Beginner";
  return difficulty.charAt(0).toUpperCase() + difficulty.slice(1);
};

const roleLabel = (role: string) => {
  if (!role) return "Learner";
  return role.charAt(0).toUpperCase() + role.slice(1);
};

const statusLabel = (status: string) => {
  if (!status) return "Assigned";
  return status.replace("_", " ").replace(/\b\w/g, (char) => char.toUpperCase());
};

const formatJoinDate = (value?: string | null) => {
  if (!value) return "Recently joined";

  try {
    return new Intl.DateTimeFormat("en-US", {
      month: "long",
      year: "numeric",
    }).format(new Date(value));
  } catch {
    return "Recently joined";
  }
};

const getInitials = (profile: ProfileResponse | null) => {
  if (!profile) return "LO";

  const fullName = profile.full_name?.trim() || "";
  if (fullName) {
    const parts = fullName.split(/\s+/).filter(Boolean);
    if (parts.length >= 2) {
      return `${parts[0][0]}${parts[1][0]}`.toUpperCase();
    }
    if (parts.length === 1 && parts[0].length >= 2) {
      return parts[0].slice(0, 2).toUpperCase();
    }
    if (parts.length === 1) {
      return parts[0][0].toUpperCase();
    }
  }

  const emailPrefix = profile.email?.split("@", 1)[0]?.replace(/[^a-zA-Z0-9]/g, "") || "";
  if (emailPrefix.length >= 2) {
    return emailPrefix.slice(0, 2).toUpperCase();
  }
  if (emailPrefix.length === 1) {
    return emailPrefix.toUpperCase();
  }

  return "LO";
};

const getDisplayName = (profile: ProfileResponse | null) => {
  if (!profile) return "LearningOS User";

  const fullName = profile.full_name?.trim();
  if (fullName) return fullName;

  const combinedName = [profile.first_name, profile.last_name].filter(Boolean).join(" ").trim();
  if (combinedName) return combinedName;

  return profile.email || "LearningOS User";
};

export default function ProfilePage() {
  const router = useRouter();
  const [profile, setProfile] = useState<ProfileResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const loadProfile = async () => {
      setLoading(true);
      setError(null);

      try {
        const response = await getProfile();
        setProfile(response.data);
      } catch (err: unknown) {
        const detail =
          err && typeof err === "object" && "response" in err
            ? (err as { response?: { data?: { detail?: string } } }).response?.data?.detail
            : undefined;
        const message =
          err instanceof Error
            ? err.message
            : undefined;
        setError(
          detail ||
          message ||
          "Failed to load profile"
        );
      } finally {
        setLoading(false);
      }
    };

    loadProfile();
  }, []);

  const initials = useMemo(() => {
    return getInitials(profile);
  }, [profile]);

  const displayName = useMemo(() => {
    return getDisplayName(profile);
  }, [profile]);

  const levelBadgeLabel = useMemo(() => {
    return profile ? levelLabel(profile.current_level) : "";
  }, [profile]);

  const difficultyBadgeLabel = useMemo(() => {
    return profile ? difficultyLabel(profile.difficulty) : "";
  }, [profile]);

  if (loading) {
    return (
      <main className="min-h-screen bg-background px-6 py-8 md:px-8">
        <div className="mx-auto max-w-7xl">
          <Card className="rounded-3xl border-border bg-card shadow-sm dark:border-slate-800 dark:bg-black">
            <CardContent className="p-6 text-muted-foreground">
              Loading your profile...
            </CardContent>
          </Card>
        </div>
      </main>
    );
  }

  if (error || !profile) {
    return (
      <main className="min-h-screen bg-background px-6 py-8 md:px-8">
        <div className="mx-auto max-w-7xl">
          <Card className="rounded-3xl border-red-200 bg-card shadow-sm dark:border-red-900 dark:bg-black">
            <CardContent className="p-6">
              <p className="text-sm text-red-600">
                {error || "Failed to load profile"}
              </p>
            </CardContent>
          </Card>
        </div>
      </main>
    );
  }

  return (
    <main className="min-h-screen bg-background px-6 py-8 md:px-8">
      <div className="mx-auto max-w-7xl space-y-6">
        <Card className="overflow-hidden rounded-3xl border-border bg-card shadow-sm dark:border-slate-800 dark:bg-black">
          <div className="bg-card dark:bg-black">
            <CardContent className="p-0">
              <div className="grid gap-8 px-6 py-8 text-foreground md:px-8 lg:grid-cols-[1.4fr_0.9fr]">
                <div className="space-y-6">
                  <div className="flex items-start gap-4">
                    <Avatar className="h-20 w-20 border border-border bg-muted dark:border-slate-800 dark:bg-slate-950" size="lg">
                      <AvatarFallback className="bg-muted text-lg font-semibold text-foreground dark:bg-slate-950">
                        {initials}
                      </AvatarFallback>
                    </Avatar>

                    <div className="space-y-3">
                      <div>
                        <p className="text-sm font-medium uppercase tracking-[0.25em] text-muted-foreground">
                          Learning Profile
                        </p>
                        <h1 className="mt-2 text-4xl font-bold tracking-tight">
                          {displayName}
                        </h1>
                      </div>

                      <div className="flex flex-wrap items-center gap-2">
                        <Badge className="rounded-full bg-muted text-foreground hover:bg-muted dark:bg-slate-950 dark:hover:bg-slate-950">
                          {roleLabel(profile.role)}
                        </Badge>
                        <Badge className="rounded-full bg-muted text-foreground hover:bg-muted dark:bg-slate-950 dark:hover:bg-slate-950">
                          {levelBadgeLabel}
                        </Badge>
                        {difficultyBadgeLabel !== levelBadgeLabel && (
                          <Badge className="rounded-full bg-muted text-foreground hover:bg-muted dark:bg-slate-950 dark:hover:bg-slate-950">
                            {difficultyBadgeLabel}
                          </Badge>
                        )}
                      </div>
                    </div>
                  </div>

                  <div className="grid gap-3 sm:grid-cols-2">
                    <InfoChip
                      icon={<Mail className="h-4 w-4" />}
                      label="Email"
                      value={profile.email}
                    />
                    <InfoChip
                      icon={<Clock3 className="h-4 w-4" />}
                      label="Joined"
                      value={formatJoinDate(profile.joined_at)}
                    />
                  </div>
                </div>

                <div className="rounded-3xl border border-border bg-muted/40 p-5 dark:border-slate-800 dark:bg-slate-950">
                  <div className="flex items-start justify-between gap-4">
                    <div>
                      <p className="text-sm text-muted-foreground">Current lesson</p>
                      <h2 className="mt-2 text-2xl font-bold text-foreground">
                        {profile.current_lesson?.title || "Your next lesson is ready"}
                      </h2>
                    </div>
                    <Sparkles className="h-5 w-5 text-muted-foreground" />
                  </div>

                  <div className="mt-6 space-y-3">
                    <div className="flex items-center justify-between text-sm text-muted-foreground">
                      <span>
                        {profile.current_lesson
                          ? statusLabel(profile.current_lesson.status)
                          : "Assigned"}
                      </span>
                      <span>{profile.current_lesson?.progress ?? 0}%</span>
                    </div>
                    <Progress
                      value={profile.current_lesson?.progress ?? 0}
                      className="h-2.5 bg-background/70"
                    />
                  </div>

                  <Button
                    className="mt-6 w-full rounded-xl bg-blue-600 text-white hover:bg-blue-700"
                    onClick={() =>
                      profile.current_lesson?.source
                        ? router.push(
                            `/lesson-details?source=${encodeURIComponent(
                              profile.current_lesson.source
                            )}`
                          )
                        : router.push("/lessons")
                    }
                  >
                    Continue Learning
                    <ArrowRight className="ml-2 h-4 w-4" />
                  </Button>
                </div>
              </div>
            </CardContent>
          </div>
        </Card>

        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
          <ProfileStatCard
            icon={<BookCheck className="h-5 w-5 text-blue-600" />}
            title="Lessons Completed"
            value={profile.stats.lessons_completed.toString()}
            caption="Finished and ready for review"
          />
          <ProfileStatCard
            icon={<GraduationCap className="h-5 w-5 text-amber-500" />}
            title="In Progress"
            value={profile.stats.lessons_in_progress.toString()}
            caption="Lessons you can resume now"
          />
          <ProfileStatCard
            icon={<Trophy className="h-5 w-5 text-emerald-600" />}
            title="Average Quiz Score"
            value={`${profile.stats.average_quiz_score}%`}
            caption="Your running assessment average"
          />
          <ProfileStatCard
            icon={<Award className="h-5 w-5 text-violet-600" />}
            title="Last Quiz Result"
            value={`${profile.stats.last_quiz_score}%`}
            caption="Most recent checkpoint performance"
          />
        </div>

        <div className="grid gap-5 xl:grid-cols-[1.3fr_0.9fr]">
          <Card className="rounded-3xl border-border bg-card shadow-sm dark:border-slate-800 dark:bg-black">
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
                  {roleLabel(profile.role)} Path
                </Badge>
              </div>

              <div className="mt-6 grid gap-4 md:grid-cols-2">
                <SnapshotBlock
                  label="Current Level"
                  value={levelLabel(profile.current_level)}
                  helper="Adapted from your questionnaire and progress"
                />
                <SnapshotBlock
                  label="Assigned Lessons"
                  value={profile.stats.lessons_assigned.toString()}
                  helper="Queued and ready to start"
                />
                
              </div>
            </CardContent>
          </Card>

          <Card className="rounded-3xl border-border bg-card shadow-sm dark:border-slate-800 dark:bg-black">
            <CardContent className="p-6">
              <div>
                <h2 className="text-2xl font-bold text-foreground">
                  Next Move
                </h2>
                <p className="mt-1 text-sm text-muted-foreground">
                  The fastest way to keep your momentum.
                </p>
              </div>

              <div className="mt-6 rounded-2xl bg-muted/40 p-4 dark:bg-slate-950">
                <p className="text-sm font-medium text-muted-foreground">
                  Recommended focus
                </p>
                <h3 className="mt-2 text-xl font-bold text-foreground">
                  {profile.current_lesson?.title || "Open your assigned lessons"}
                </h3>
                <p className="mt-2 text-sm leading-6 text-muted-foreground">
                  {profile.current_lesson
                    ? `You already have progress here. Jump back in and finish the next part while it's still fresh.`
                    : "Your lesson path is ready. Open the lessons page and start the next recommended topic."}
                </p>
              </div>

              <Button
                className="mt-5 w-full rounded-xl bg-blue-600 text-white hover:bg-blue-700"
                onClick={() =>
                  profile.current_lesson?.source
                    ? router.push(
                        `/lesson-details?source=${encodeURIComponent(
                          profile.current_lesson.source
                        )}`
                      )
                    : router.push("/dashboard")
                }
              >
                Open Lesson
              </Button>
            </CardContent>
          </Card>
        </div>
      </div>
    </main>
  );
}

function InfoChip({
  icon,
  label,
  value,
}: {
  icon: ReactNode;
  label: string;
  value: string;
}) {
  return (
    <div className="rounded-2xl border border-slate-200 bg-slate-50 px-4 py-3 dark:border-slate-800 dark:bg-slate-950">
      <div className="flex items-center gap-2 text-sm text-slate-400">
        {icon}
        <span>{label}</span>
      </div>
      <p className="mt-2 text-base font-medium text-slate-900 dark:text-slate-100">{value}</p>
    </div>
  );
}

function ProfileStatCard({
  icon,
  title,
  value,
  caption,
}: {
  icon: ReactNode;
  title: string;
  value: string;
  caption: string;
}) {
  return (
    <Card className="rounded-2xl border-slate-200 bg-white shadow-sm dark:border-slate-800 dark:bg-black">
      <CardContent className="p-5">
        <div className="flex items-center justify-between gap-3">
          <p className="text-sm text-slate-400">{title}</p>
          <div className="rounded-xl bg-slate-50 p-2 dark:bg-slate-950">{icon}</div>
        </div>
        <h3 className="mt-4 text-4xl font-bold text-slate-900 dark:text-slate-100">{value}</h3>
        <p className="mt-2 text-sm leading-6 text-slate-400">{caption}</p>
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
