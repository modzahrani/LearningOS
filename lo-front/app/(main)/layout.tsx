"use client";

import { AppSidebar } from "@/components/app-sidebar";
import { ThemeToggle } from "@/components/theme-toggle";
import {
  SidebarInset,
  SidebarProvider,
  SidebarTrigger,
} from "@/components/ui/sidebar";

export default function MainLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <SidebarProvider>
      <AppSidebar />

      <SidebarInset>
        <header className="flex h-16 items-center justify-between border-b border-border bg-background px-4">
          <SidebarTrigger />
          <ThemeToggle />
        </header>

        <div className="min-h-[calc(100vh-64px)] bg-background">
          {children}
        </div>
      </SidebarInset>
    </SidebarProvider>
  );
}
