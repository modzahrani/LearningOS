"use client";

import { useEffect, useMemo, useState } from "react";
import {
  Bot,
  BookOpen,
  LayoutDashboard,
  LogOut,
  User,
} from "lucide-react";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";

import { getProfile, logout } from "@/api/userProvider";
import type { ProfileResponse } from "@/api/userProvider";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarGroupContent,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
} from "@/components/ui/sidebar";

const items = [
  {
    title: "Dashboard",
    url: "/dashboard",
    icon: LayoutDashboard,
  },
  {
    title: "Lessons",
    url: "/lessons",
    icon: BookOpen,
  },
  {
    title: "Profile",
    url: "/profile",
    icon: User,
  },
  {
    title: "AI Chatbot",
    url: "/chatbot",
    icon: Bot,
  },
];

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

export function AppSidebar() {
  const pathname = usePathname();
  const router = useRouter();
  const [profile, setProfile] = useState<ProfileResponse | null>(null);
  const [isSigningOut, setIsSigningOut] = useState(false);

  useEffect(() => {
    const loadProfile = async () => {
      try {
        const response = await getProfile();
        setProfile(response.data);
      } catch {
        setProfile(null);
      }
    };

    loadProfile();
  }, []);

  const initials = useMemo(() => {
    return getInitials(profile);
  }, [profile]);

  const roleLabel = useMemo(() => {
    if (!profile?.role) return "Member";
    return profile.role.charAt(0).toUpperCase() + profile.role.slice(1);
  }, [profile]);

  const handleSignOut = async () => {
    if (isSigningOut) return;

    setIsSigningOut(true);

    try {
      await logout();
    } catch {
      // Clear local session even if the server logout request fails.
    } finally {
      router.replace("/login");
      setIsSigningOut(false);
    }
  };

  return (
    <Sidebar className="dark:border-slate-800 dark:bg-black">
      <SidebarHeader className="border-b dark:border-slate-800">
        <div className="flex items-center gap-3 px-2 py-3">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-blue-600 text-white">

            {/*TODO: change BookOpen to a logo */}
            <BookOpen className="h-4 w-4" />
          </div>

          <div className="flex flex-col">
            <span className="text-lg font-bold text-slate-900 dark:text-slate-100">LearningOS</span>
            <span className="text-xs text-slate-400 dark:text-slate-500">
              Personalized Learning
            </span>
          </div>
        </div>
      </SidebarHeader>

      <SidebarContent>
        <SidebarGroup>
          <SidebarGroupContent>
            <SidebarMenu>
              {items.map((item) => {
                const isActive = pathname === item.url;

                return (
                  <SidebarMenuItem key={item.title}>
                    <SidebarMenuButton asChild isActive={isActive}>
                      <Link href={item.url}>
                        <item.icon className="h-4 w-4" />
                        <span>{item.title}</span>
                      </Link>
                    </SidebarMenuButton>
                  </SidebarMenuItem>
                );
              })}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>

      <SidebarFooter className="border-t dark:border-slate-800">
        <div className="space-y-3 px-2 py-3">
          <div className="flex items-center gap-3">
            <Avatar className="h-10 w-10">
              <AvatarFallback className="dark:bg-slate-950 dark:text-slate-100">{initials}</AvatarFallback>
            </Avatar>

            <div className="flex min-w-0 flex-col">
              <span className="truncate text-sm font-semibold text-slate-900 dark:text-slate-100">
                {profile?.full_name || "LearningOS User"}
              </span>
              <span className="text-xs text-slate-400 dark:text-slate-500">{roleLabel}</span>
            </div>
          </div>

          <button
            type="button"
            onClick={handleSignOut}
            disabled={isSigningOut}
            className="flex w-full items-center justify-center gap-2 rounded-xl bg-blue-600 px-3 py-2 text-sm font-medium text-white transition hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-60"
          >
            <LogOut className="h-4 w-4" />
            {isSigningOut ? "Signing out..." : "Sign out"}
          </button>
        </div>
      </SidebarFooter>
    </Sidebar>
  );
}
