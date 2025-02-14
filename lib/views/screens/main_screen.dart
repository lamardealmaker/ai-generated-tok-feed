import 'dart:ui';
import 'package:flutter/material.dart';
import 'package:get/get.dart';
import '../../controllers/video_controller.dart';
import '../../constants.dart';
import 'home_screen.dart';
import 'favorites_screen.dart';
import 'profile_screen.dart';

class MainScreen extends StatefulWidget {
  const MainScreen({super.key});

  @override
  State<MainScreen> createState() => _MainScreenState();
}

class _MainScreenState extends State<MainScreen> {
  int _selectedIndex = 0;

  Widget _getScreen(int index) {
    switch (index) {
      case 0:
        return const HomeScreen();
      case 1:
        return const FavoritesScreen();
      case 2:
        return ProfileScreen(); 
      default:
        return const HomeScreen();
    }
  }

  final VideoController videoController = Get.find<VideoController>();

  void _onItemTapped(int index) {
    // Pause video if leaving home screen
    if (_selectedIndex == 0 && index != 0) {
      final currentVideoId = videoController.videos[videoController.currentVideoIndex.value].id;
      final currentController = videoController.getController(currentVideoId);
      currentController?.pause();
    }

    setState(() {
      _selectedIndex = index;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.background,
      body: _getScreen(_selectedIndex),
      bottomNavigationBar: Container(
        decoration: BoxDecoration(
          color: AppColors.darkGrey.withOpacity(0.95),
          border: Border(
            top: BorderSide(
              color: AppColors.accent.withOpacity(0.2),
              width: 1,
            ),
          ),
          boxShadow: [
            BoxShadow(
              color: Colors.black.withOpacity(0.2),
              blurRadius: 8,
              offset: const Offset(0, -2),
            ),
          ],
        ),
        child: ClipRRect(
          child: BackdropFilter(
            filter: ImageFilter.blur(sigmaX: 10, sigmaY: 10),
            child: BottomNavigationBar(
              elevation: 0,
              backgroundColor: Colors.transparent,
              selectedItemColor: AppColors.accent,
              unselectedItemColor: AppColors.textSecondary,
              selectedLabelStyle: const TextStyle(
                fontWeight: FontWeight.w600,
                fontSize: 12,
              ),
              unselectedLabelStyle: const TextStyle(
                fontWeight: FontWeight.normal,
                fontSize: 12,
              ),
              type: BottomNavigationBarType.fixed,
              currentIndex: _selectedIndex,
              onTap: _onItemTapped,
              items: [
                BottomNavigationBarItem(
                  icon: Icon(
                    Icons.home_outlined,
                    size: 24,
                    color: AppColors.textSecondary,
                  ),
                  activeIcon: Container(
                    decoration: BoxDecoration(
                      color: AppColors.accent.withOpacity(0.1),
                      borderRadius: BorderRadius.circular(8),
                    ),
                    padding: const EdgeInsets.all(8),
                    child: Icon(
                      Icons.home_rounded,
                      size: 24,
                      color: AppColors.accent,
                    ),
                  ),
                  label: 'Home',
                ),
                BottomNavigationBarItem(
                  icon: Icon(
                    Icons.bookmark_outline,
                    size: 24,
                    color: AppColors.textSecondary,
                  ),
                  activeIcon: Container(
                    decoration: BoxDecoration(
                      color: AppColors.accent.withOpacity(0.1),
                      borderRadius: BorderRadius.circular(8),
                    ),
                    padding: const EdgeInsets.all(8),
                    child: Icon(
                      Icons.bookmark_rounded,
                      size: 24,
                      color: AppColors.accent,
                    ),
                  ),
                  label: 'Favorites',
                ),
                BottomNavigationBarItem(
                  icon: Icon(
                    Icons.person_outline,
                    size: 24,
                    color: AppColors.textSecondary,
                  ),
                  activeIcon: Container(
                    decoration: BoxDecoration(
                      color: AppColors.accent.withOpacity(0.1),
                      borderRadius: BorderRadius.circular(8),
                    ),
                    padding: const EdgeInsets.all(8),
                    child: Icon(
                      Icons.person_rounded,
                      size: 24,
                      color: AppColors.accent,
                    ),
                  ),
                  label: 'Profile',
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
