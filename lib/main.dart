import 'package:flutter/material.dart';
import 'package:firebase_core/firebase_core.dart';
import 'package:get/get.dart';
import 'constants.dart';
import 'views/screens/auth/login_screen.dart';
import 'views/screens/auth/email_verification_screen.dart';
import 'views/screens/main_screen.dart';
import 'views/screens/home_screen.dart';
import 'views/screens/favorites_screen.dart';
import 'views/screens/profile_screen.dart';
import 'controllers/auth_controller.dart';
import 'controllers/video_controller.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await Firebase.initializeApp();
  
  // Initialize Controllers
  Get.put(AuthController());
  Get.put(VideoController());
  
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return GetMaterialApp(
      title: AppConstants.appName,
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        useMaterial3: true,
        primaryColor: AppColors.primary,
        scaffoldBackgroundColor: AppColors.background,
        colorScheme: ColorScheme.dark(
          primary: AppColors.accent,
          secondary: AppColors.save,
          background: AppColors.background,
          surface: AppColors.darkGrey,
          onSurface: AppColors.textPrimary,
          onBackground: AppColors.textPrimary,
          onPrimary: AppColors.buttonText,
          onSecondary: AppColors.white,
        ),
        textTheme: const TextTheme(
          displayLarge: TextStyle(color: AppColors.textPrimary, fontWeight: FontWeight.bold),
          displayMedium: TextStyle(color: AppColors.textPrimary, fontWeight: FontWeight.bold),
          displaySmall: TextStyle(color: AppColors.textPrimary, fontWeight: FontWeight.bold),
          headlineLarge: TextStyle(color: AppColors.textPrimary, fontWeight: FontWeight.w600),
          headlineMedium: TextStyle(color: AppColors.textPrimary, fontWeight: FontWeight.w600),
          headlineSmall: TextStyle(color: AppColors.textPrimary, fontWeight: FontWeight.w600),
          titleLarge: TextStyle(color: AppColors.textPrimary, fontWeight: FontWeight.w600),
          titleMedium: TextStyle(color: AppColors.textPrimary),
          titleSmall: TextStyle(color: AppColors.textSecondary),
          bodyLarge: TextStyle(color: AppColors.textPrimary),
          bodyMedium: TextStyle(color: AppColors.textPrimary),
          bodySmall: TextStyle(color: AppColors.textSecondary),
          labelLarge: TextStyle(color: AppColors.textPrimary, fontWeight: FontWeight.w600),
          labelMedium: TextStyle(color: AppColors.textPrimary),
          labelSmall: TextStyle(color: AppColors.textSecondary),
        ),
        iconTheme: const IconThemeData(
          color: AppColors.textPrimary,
          size: AppTheme.iconSize_md,
        ),
        elevatedButtonTheme: ElevatedButtonThemeData(
          style: ElevatedButton.styleFrom(
            backgroundColor: AppColors.buttonBackground,
            foregroundColor: AppColors.buttonText,
            elevation: 0,
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(12),
            ),
          ),
        ),
      ),
      initialRoute: '/login',
      getPages: [
        GetPage(name: '/login', page: () => const LoginScreen()),
        GetPage(name: '/verify-email', page: () => const EmailVerificationScreen()),
        GetPage(name: '/home', page: () => const MainScreen()),
        GetPage(name: '/home-screen', page: () => const HomeScreen()),
        GetPage(name: '/favorites', page: () => const FavoritesScreen()),
        GetPage(name: '/profile', page: () => ProfileScreen()),
      ],
    );
  }
}
