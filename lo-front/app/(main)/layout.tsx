"use client";

import { AppSidebar } from "@/components/app-sidebar";
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
        <header className="flex h-16 items-center border-b bg-white px-4">
          <SidebarTrigger />
        </header>

        <div className="min-h-[calc(100vh-64px)] bg-[#f4f7fb]">
          {children}
        </div>
      </SidebarInset>
    </SidebarProvider>
  );
}
