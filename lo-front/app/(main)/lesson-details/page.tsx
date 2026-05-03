'use client'

import React, { useState } from 'react';
import { 
  Play, 
  CheckCircle2, 
  Lock, 
  Clock, 
  Signal, 
  Star, 
  ChevronRight, 
  Settings, 
  Maximize, 
  Subtitles 
} from 'lucide-react';

export default function LessonPage() {
  const [activeTab, setActiveTab] = useState('Overview');

  return (
    <div className="max-w-7xl mx-auto p-4 md:p-8 animate-in fade-in duration-500">
      {/* Breadcrumbs */}
      <nav className="flex items-center gap-2 text-xs text-slate-400 mb-6">
        <span>Courses</span> <ChevronRight size={10} />
        <span>Data Science Track</span> <ChevronRight size={10} />
        <span className="text-slate-600 font-medium">Introduction to Python Data Structures</span>
      </nav>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-10">
        
        {/* Left Column: Video & Info (Col 8) */}
        <div className="lg:col-span-8">
          
          {/* Video Player */}
          <div className="group relative aspect-video bg-slate-900 rounded-2xl overflow-hidden shadow-2xl mb-8">
            <div className="absolute inset-0 flex items-center justify-center">
              <button className="w-16 h-16 bg-blue-600 hover:bg-blue-500 rounded-full flex items-center justify-center text-white transition-all hover:scale-110 shadow-lg">
                <Play fill="currentColor" size={24} className="ml-1" />
              </button>
            </div>
            
            {/* Player Controls Overlay */}
            <div className="absolute bottom-0 w-full p-4 bg-gradient-to-t from-black/80 to-transparent flex items-center gap-4 text-white">
              <Play size={14} fill="currentColor" />
              <div className="flex-1 h-1 bg-white/20 rounded-full relative">
                <div className="absolute w-1/3 h-full bg-blue-500 rounded-full" />
              </div>
              <span className="text-[10px] font-mono">04:20 / 12:45</span>
              <Subtitles size={14} className="opacity-70 hover:opacity-100 cursor-pointer" />
              <Settings size={14} className="opacity-70 hover:opacity-100 cursor-pointer" />
              <Maximize size={14} className="opacity-70 hover:opacity-100 cursor-pointer" />
            </div>
          </div>

          {/* Lesson Header */}
          <div className="flex flex-wrap items-center gap-4 text-[10px] font-bold uppercase tracking-widest mb-4">
            <span className="bg-blue-100 text-blue-700 px-2 py-1 rounded">Lesson 4</span>
            <span className="text-slate-400 flex items-center gap-1"><Clock size={12} /> 12 min</span>
            <span className="text-slate-400 flex items-center gap-1"><Signal size={12} /> Intermediate</span>
            <span className="text-amber-500 flex items-center gap-1">
              <Star size={12} fill="currentColor" /> 4.8 
              <span className="text-slate-300 font-normal lowercase tracking-normal">(12k reviews)</span>
            </span>
          </div>

          <h1 className="text-3xl font-bold text-slate-800 mb-4 tracking-tight">
            Introduction to Python Data Structures
          </h1>
          <p className="text-slate-500 leading-relaxed mb-8">
            Master lists, dictionaries, tuples, and sets to manage data efficiently in your applications. 
            This foundational lesson sets the stage for advanced algorithm design.
          </p>

          {/* Tabs Navigation */}
          <div className="flex gap-8 border-b border-slate-200 mb-8 overflow-x-auto">
            {['Overview', 'Resources', 'Q&A', 'Transcript', 'Reviews'].map((tab) => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className={`pb-4 text-sm font-semibold transition-all relative ${
                  activeTab === tab ? 'text-blue-600' : 'text-slate-400 hover:text-slate-600'
                }`}
              >
                {tab}
                {tab === 'Resources' && <span className="ml-1 text-[10px] bg-slate-100 px-1.5 py-0.5 rounded-full">3</span>}
                {activeTab === tab && <div className="absolute bottom-0 left-0 w-full h-0.5 bg-blue-600" />}
              </button>
            ))}
          </div>

          {/* Tab Content */}
          <div className="space-y-4">
            <h2 className="text-xl font-bold text-slate-800">About this lesson</h2>
            <div className="text-slate-500 text-sm leading-7">
              <p>Data structures are the building blocks of efficient software. In this lesson, we will move beyond simple variables and explore how to store collections of data.</p>
            </div>
          </div>
        </div>

        {/* Right Column: Sidebar (Col 4) */}
        <div className="lg:col-span-4 space-y-6">
          
          {/* Progress Card */}
          <div className="bg-white p-6 rounded-2xl border border-slate-100 shadow-sm">
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-sm font-bold text-slate-800">Course Progress</h3>
              <span className="text-xs text-blue-600 font-bold">45%</span>
            </div>
            <div className="h-1.5 w-full bg-slate-100 rounded-full overflow-hidden mb-3">
              <div className="h-full bg-blue-600 rounded-full" style={{ width: '45%' }} />
            </div>
            <p className="text-[10px] text-slate-400">12 of 28 lessons completed</p>
          </div>

          {/* Playlist */}
          <div className="bg-white rounded-2xl border border-slate-100 shadow-sm overflow-hidden">
            <div className="p-5 border-b border-slate-50">
              <h3 className="text-sm font-bold text-slate-800">Course Content</h3>
            </div>
            
            <div className="divide-y divide-slate-50">
              <PlaylistItem title="1. Introduction to the course" duration="5 min" status="completed" />
              <PlaylistItem title="2. Setting up your environment" duration="15 min" status="completed" />
              <PlaylistItem title="3. Python Basics: Variables" duration="25 min" status="completed" />
              <PlaylistItem title="4. Data Structures: Lists..." duration="12 min" status="active" />
              <PlaylistItem title="5. Data structures: Dictionaries" duration="18 min" status="locked" />
            </div>
          </div>

        </div>
      </div>
    </div>
  );
}

// Sub-component for the Lesson Items
function PlaylistItem({ title, duration, status }: { title: string, duration: string, status: 'completed' | 'active' | 'locked' }) {
  const isActive = status === 'active';
  const isLocked = status === 'locked';

  return (
    <div className={`flex gap-4 p-4 transition-colors cursor-pointer ${isActive ? 'bg-blue-50/50 border-l-4 border-blue-600' : 'hover:bg-slate-50'}`}>
      <div className={`mt-0.5 ${isActive ? 'text-blue-600' : isLocked ? 'text-slate-300' : 'text-green-500'}`}>
        {status === 'completed' && <CheckCircle2 size={16} />}
        {status === 'active' && <Play size={16} fill="currentColor" />}
        {status === 'locked' && <Lock size={14} />}
      </div>
      <div className="flex-1">
        <p className={`text-xs font-bold ${isActive ? 'text-blue-900' : isLocked ? 'text-slate-400' : 'text-slate-700'}`}>
          {title}
        </p>
        <p className={`text-[10px] mt-1 ${isActive ? 'text-blue-500 font-semibold' : 'text-slate-400'}`}>
          {duration} {isActive && "• Now Playing"}
        </p>
      </div>
      {isActive && <div className="w-1.5 h-1.5 rounded-full bg-blue-600 mt-1" />}
    </div>
  );
}
