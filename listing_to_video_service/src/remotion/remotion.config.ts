import { Config } from 'remotion';

Config.Rendering.setImageFormat('jpeg');
Config.Output.setCodec('h264');
Config.Rendering.setQuality(80);

Config.Bundling.overrideWebpackConfig((currentConfiguration) => {
  return {
    ...currentConfiguration,
    module: {
      ...currentConfiguration.module,
      rules: [
        ...(currentConfiguration.module?.rules ?? []),
        {
          test: /\.(woff|woff2|eot|ttf|otf)$/,
          type: 'asset/resource',
        },
      ],
    },
  };
});

Config.Preview.setMaxTimelineTracks(20);

Config.Output.setPixelFormat('yuv420p');
Config.Output.setCodec('h264');
Config.Output.setImageSequence(false);
Config.Output.setScale(1);

// Set dimensions for 9:16 aspect ratio (1080x1920)
Config.Rendering.setImageFormat('jpeg');
Config.Rendering.setQuality(80);
Config.Rendering.setDefaultProps({
  width: 1080,
  height: 1920,
  fps: 30,
});
