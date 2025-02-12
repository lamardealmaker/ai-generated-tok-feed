import { useCallback, useEffect, useState } from 'react';
import {
  AbsoluteFill,
  Audio,
  Sequence,
  spring,
  useCurrentFrame,
  useVideoConfig,
  interpolate,
  random,
} from 'remotion';
import styled from 'styled-components';

const Container = styled(AbsoluteFill)`
  background-color: black;
  display: flex;
  flex-direction: column;
`;

const ImageContainer = styled.div`
  flex: 1;
  position: relative;
  overflow: hidden;
`;

const StyledImage = styled.img`
  width: 100%;
  height: 100%;
  object-fit: cover;
`;

const CaptionContainer = styled.div`
  position: absolute;
  bottom: 100px;
  left: 20px;
  right: 20px;
  color: white;
  font-family: 'SF Pro Display', sans-serif;
  text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
`;

interface PropertyVideoProps {
  propertyData: {
    address: string;
    price: number;
    beds: number;
    baths: number;
    sqft: number;
    images: string[];
  };
  script: string;
  audioPath: string;
  style: 'parallax' | 'kenBurns' | 'splitScreen' | '3dRotation' | 'pictureInPicture';
}

const VIDEO_DURATION = 900; // 30 seconds at 30fps
const TRANSITION_DURATION = 30;

const applyImageEffect = (
  frame: number,
  style: PropertyVideoProps['style'],
  index: number
) => {
  const progress = spring({
    frame,
    fps: 30,
    config: {
      damping: 100,
      stiffness: 200,
    },
  });

  switch (style) {
    case 'parallax':
      return {
        transform: `scale(1.1) translateX(${interpolate(
          progress,
          [0, 1],
          [0, -50]
        )}px)`,
      };
    case 'kenBurns':
      return {
        transform: `scale(${interpolate(progress, [0, 1], [1, 1.2])})`,
      };
    case '3dRotation':
      return {
        transform: `perspective(1000px) rotateY(${interpolate(
          progress,
          [0, 1],
          [0, 15]
        )}deg)`,
      };
    default:
      return {};
  }
};

const WordAnimation: React.FC<{ word: string; delay: number }> = ({
  word,
  delay,
}) => {
  const frame = useCurrentFrame();
  const progress = spring({
    frame: frame - delay,
    fps: 30,
    config: {
      damping: 100,
      stiffness: 200,
    },
  });

  return (
    <span
      style={{
        display: 'inline-block',
        opacity: progress,
        transform: `translateY(${interpolate(
          progress,
          [0, 1],
          [20, 0]
        )}px)`,
        marginRight: '0.3em',
      }}
    >
      {word}
    </span>
  );
};

export const PropertyVideo: React.FC<PropertyVideoProps> = ({
  propertyData,
  script,
  audioPath,
  style,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const words = script.split(' ');

  const renderImage = useCallback(
    (imageUrl: string, index: number) => {
      const startFrame = index * (VIDEO_DURATION / propertyData.images.length);
      const endFrame = startFrame + VIDEO_DURATION / propertyData.images.length;

      return (
        <Sequence from={startFrame} durationInFrames={endFrame - startFrame}>
          <ImageContainer>
            <StyledImage
              src={imageUrl}
              style={applyImageEffect(frame - startFrame, style, index)}
            />
          </ImageContainer>
        </Sequence>
      );
    },
    [frame, style]
  );

  return (
    <Container>
      {propertyData.images.map((image, index) => renderImage(image, index))}
      
      <Audio src={audioPath} />

      <CaptionContainer>
        {words.map((word, i) => (
          <WordAnimation
            key={i}
            word={word}
            delay={i * 3}
          />
        ))}
      </CaptionContainer>

      {/* Property Stats Overlay */}
      <Sequence from={0}>
        <div
          style={{
            position: 'absolute',
            top: 20,
            right: 20,
            color: 'white',
            fontSize: 24,
            textAlign: 'right',
          }}
        >
          <div>${propertyData.price.toLocaleString()}</div>
          <div>{propertyData.beds} beds • {propertyData.baths} baths</div>
          <div>{propertyData.sqft.toLocaleString()} sqft</div>
        </div>
      </Sequence>
    </Container>
  );
};

export default PropertyVideo;
