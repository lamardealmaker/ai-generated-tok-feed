from typing import Dict, Any, List, Optional, Callable
import asyncio
from datetime import datetime
import uuid
import os
from src.utils.logging import setup_logger
from ..agents.script_agent import ScriptAgent
from ..agents.pacing_agent import PacingAgent
from ..agents.effect_agent import EffectAgent
from ..tools.video_db_tool import VideoDBTool
from ..tools.video_renderer import VideoRenderer
from ..tools.music_tool import MusicTool
from ..tools.voiceover_tool import VoiceoverTool
from .task import Task

class ReasoningEngine:
    """Core reasoning and workflow orchestration engine."""
    
    def __init__(self):
        self.logger = setup_logger("reasoning_engine")
        self.logger.info("Initializing ReasoningEngine...")
        
        self.script_agent = ScriptAgent()
        self.pacing_agent = PacingAgent()
        self.effect_agent = EffectAgent()
        self.video_db_tool = VideoDBTool()
        self.video_renderer = VideoRenderer()
        self.music_tool = MusicTool()
        self.voiceover_tool = VoiceoverTool()
        
        # Create audio directory if it doesn't exist
        self.audio_dir = os.path.join(os.getcwd(), "assets/audio")
        os.makedirs(os.path.join(self.audio_dir, "voiceover"), exist_ok=True)
        os.makedirs(os.path.join(self.audio_dir, "music"), exist_ok=True)
        
        self.logger.info("All components initialized successfully")
        
        self.workflow_hooks: Dict[str, List[Callable]] = {
            "pre_generation": [],
            "post_script": [],
            "post_effects": [],
            "post_generation": []
        }
        
    async def generate_video(
        self,
        listing_data: Dict[str, Any],
        image_urls: List[str],
        style: str = "modern",
        duration: float = 30.0,
        session_id: Optional[str] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> Task:
        """
        Generate a complete real estate video.
        
        Args:
            property_data: Property listing data
            style: Video style to use
            duration: Target video duration
            session_id: Optional session ID for tracking
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary containing video metadata and output paths
        """
        try:
            # Run pre-generation hooks
            await self._run_hooks("pre_generation", listing_data)
            
            # 1. Generate script
            self.logger.info(f"Generating script for property {listing_data.get('id', 'unknown')}")
            if progress_callback:
                progress_callback(0.1, "Generating script")
                
            script = self.script_agent.generate_script(
                property_data=listing_data,
                style=style,
                duration=duration
            )
            
            # Run post-script hooks
            await self._run_hooks("post_script", script)
            
            # 2. Analyze and optimize pacing
            if progress_callback:
                progress_callback(0.2, "Optimizing pacing")
                
            content_analysis = self.pacing_agent.analyze_content(script["segments"])
            pacing = self.pacing_agent.get_optimal_pacing(
                content_analysis=content_analysis,
                target_duration=duration,
                style=style
            )
            
            segments = self.pacing_agent.apply_pacing(
                segments=script["segments"],
                pacing=pacing,
                target_duration=duration
            )
            
            # 3. Generate effects and transitions
            if progress_callback:
                progress_callback(0.3, "Generating effects")
                
            # Get base effects for the style
            base_effects = self.video_db_tool.get_effects(
                template_name=f"{style}_standard",
                intensity=0.8  # Slightly reduced intensity for better results
            )
            
            # Get transitions (one less than segments since we don't need transition for first)
            transitions = self.video_db_tool.get_transitions(
                template_name=f"{style}_standard",
                count=len(segments) - 1,
                variation=0.2
            )
            
            # Process each segment with effects and transitions
            processed_segments = []
            for i, segment in enumerate(segments):
                # Don't add transition for first segment
                transition = None if i == 0 else transitions[i-1]
                
                # Process segment
                processed = self.video_db_tool.process_segment(
                    segment=segment,
                    effects=base_effects,
                    transition=transition
                )
                processed_segments.append(processed)
            
            # Replace segments with processed ones
            segments = processed_segments
                
            # Run post-effects hooks
            await self._run_hooks("post_effects", segments)
            
            # 4. Generate voiceovers for each segment
            if progress_callback:
                progress_callback(0.4, "Generating voiceover")
                
            for segment in segments:
                if "text" in segment:
                    voiceover = self.voiceover_tool.generate_voiceover(
                        text=segment["text"],
                        style=style,
                        output_format="mp3"
                    )
                    segment["voiceover_path"] = voiceover["file_path"]
            
            # 5. Select music
            if progress_callback:
                progress_callback(0.5, "Selecting music")
                
            music = self.music_tool.select_music(
                template_name=f"{style}_balanced_standard",
                duration=duration
            )
            
            # 5. Process video
            if progress_callback:
                progress_callback(0.5, "Processing video")
                
            # 6. Finalize video
            if progress_callback:
                progress_callback(0.9, "Finalizing video")
                
            result = {
                "session_id": session_id,
                "style": style,
                "duration": duration,
                "segments": segments,  # Use processed segments
                "music": music,
                "generated_at": datetime.utcnow().isoformat(),
                "property_id": listing_data.get("id")
            }
            
            # Add image URLs to segments
            for i, segment in enumerate(segments):  # Use processed segments
                # Cycle through images if we have more segments than images
                image_index = i % len(image_urls)
                segment["image_url"] = image_urls[image_index]
            
            # Render final video
            if progress_callback:
                progress_callback(0.9, "Rendering video")
                
            output_path = self.video_renderer.render_video(
                segments=segments,  # Use processed segments
                music=music
            )
            
            result["output_path"] = output_path
            
            # Run post-generation hooks
            await self._run_hooks("post_generation", result)
            
            if progress_callback:
                progress_callback(1.0, "Complete")
                
            # Create and complete task
            task = Task(session_id or str(uuid.uuid4()))
            task.complete(result)
            return task
            
        except Exception as e:
            self.logger.error(f"Video generation failed: {str(e)}")
            # Create failed task
            task = Task(session_id or str(uuid.uuid4()))
            task.fail(str(e))
            return task
            
    def register_hook(
        self,
        hook_type: str,
        callback: Callable
    ) -> None:
        """Register a workflow hook."""
        if hook_type not in self.workflow_hooks:
            raise ValueError(f"Invalid hook type: {hook_type}")
        self.workflow_hooks[hook_type].append(callback)
        
    async def _run_hooks(
        self,
        hook_type: str,
        data: Any
    ) -> None:
        """Run all registered hooks of a given type."""
        for hook in self.workflow_hooks[hook_type]:
            try:
                if asyncio.iscoroutinefunction(hook):
                    await hook(data)
                else:
                    hook(data)
            except Exception as e:
                logger.error(f"Hook execution failed: {str(e)}")
                # Continue with other hooks
